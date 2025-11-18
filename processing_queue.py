# processing_queue.py
from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Iterable, Tuple, List, Literal

import threading
_thread_local = threading.local()

# You can still keep this around for other places, but we DO NOT strip them in canonical key now.
DEFAULT_LEADING_ARTICLES = ("the", "a", "an")

_ws = re.compile(r"\s+")
_brackets = re.compile(r"[()\[\]\{\}]")
_symbols = re.compile(r"[-*&,:;.!?/\\|+_=~`'\"<>]")


def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
    """
    One connection per thread with WAL and long busy timeout to reduce lock errors.
    """
    key = f"queue_conn__{db_path}"
    conn = getattr(_thread_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")  # 15s
        conn.execute("PRAGMA temp_store=MEMORY;")
        setattr(_thread_local, key, conn)
    return conn


def _canonical_key(s: str) -> str:
    """
    Canonical de-dupe rule:

    1) Unicode NFKC
    2) lowercase
    3) collapse spaces
    4) remove all spaces
    5) remove brackets ()[]{} and common punctuation/symbols

    Examples:
      "The Big Bang Theory (TV series)" -> "thebigbangtheorytvseries"
      "  The  Big  Bang Theory   "      -> "thebigbangtheory"
    """
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKC", s).lower()
    t = _ws.sub(" ", t).strip()
    t = t.replace(" ", "")
    t = _brackets.sub("", t)
    t = _symbols.sub("", t)
    return t


def _subject_norm(s: str) -> str:
    """
    Gentler normalization for presentation / non-unique lookups:
    - NFKC, lower, collapse spaces (keeps spaces)
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return _ws.sub(" ", s)


# ----- bootstrap / indices -----


def ensure_processed_index(conn: sqlite3.Connection):
    """
    Ensures the table used for de-duplication metadata exists.

    - processed_map: keeps an example original per canon key

    NOTE: queue table + its indices (subject_norm / subject_canon) are created
    in db_models.open_queue_db; we don't duplicate that here.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_map (
            canon_key      TEXT PRIMARY KEY,
            sample_original TEXT
        )
    """)
    conn.commit()


# ----- API -----

EnqResult = Tuple[str, int, Literal["inserted", "hop_reduced", "ignored"]]


def init_cache(conn_or_path):
    """
    Initialize any aux tables/indices needed for processed enqueue.
    Accepts either a sqlite3.Connection or a DB path.
    """
    if isinstance(conn_or_path, str):
        conn = get_thread_queue_conn(conn_or_path)
    else:
        conn = conn_or_path
    ensure_processed_index(conn)


def enqueue_subjects_processed(
    db_path_or_conn,
    items: Iterable[Tuple[str, int]],
) -> List[EnqResult]:
    """
    Enqueue with **canonical de-dupe**:

      - canonical key (subject_canon) = NFKC → lower → collapse spaces → remove spaces →
        strip brackets and common punctuation/symbols.

      - if a new variant arrives with a *lower hop*, we *lower* the hop of the existing row
        (status / retries unchanged).

      - if it’s a true duplicate with same-or-higher hop → outcome 'ignored'.

    Requires the queue table to have at least:
      subject, subject_norm, subject_canon, hop, status, retries

    Returns: list of (subject_original, kept_hop, outcome)
    """
    conn = db_path_or_conn if not isinstance(db_path_or_conn, str) else get_thread_queue_conn(db_path_or_conn)
    ensure_processed_index(conn)

    results: List[EnqResult] = []
    cur = conn.cursor()

    with conn:
        for subject, hop in items:
            if not isinstance(subject, str) or not subject.strip():
                continue

            canon = _canonical_key(subject)
            s_norm = _subject_norm(subject)

            # Keep one sample per canonical key (visibility/debug)
            cur.execute(
                "INSERT OR IGNORE INTO processed_map(canon_key, sample_original) VALUES (?, ?)",
                (canon, subject)
            )

            # Read any existing row for this canonical key to determine outcome precisely
            cur.execute("SELECT hop FROM queue WHERE subject_canon=?", (canon,))
            row = cur.fetchone()
            before_hop = row[0] if row else None

            # Upsert by canonical key; DO NOT touch status/retries if conflicting
            cur.execute(
                """
                INSERT INTO queue(subject, subject_norm, subject_canon, hop, status, retries)
                VALUES (?, ?, ?, ?, 'pending', 0)
                ON CONFLICT(subject_canon) DO UPDATE SET
                    hop = CASE WHEN excluded.hop < hop THEN excluded.hop ELSE hop END
                """,
                (subject, s_norm, canon, hop)
            )

            # Read back kept hop
            cur.execute("SELECT hop FROM queue WHERE subject_canon=?", (canon,))
            kept_hop = cur.fetchone()[0]

            if before_hop is None:
                outcome = "inserted"
            elif kept_hop < before_hop:
                outcome = "hop_reduced"
            else:
                outcome = "ignored"

            results.append((subject, kept_hop, outcome))

    return results
