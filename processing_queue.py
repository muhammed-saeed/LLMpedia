# processing_queue.py
from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Iterable, Tuple, List

import threading
_thread_local = threading.local()

DEFAULT_LEADING_ARTICLES = ("the", "a", "an")

_ws = re.compile(r"\s+")
_nonword = re.compile(r"[^a-z0-9]")

def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
    key = f"queue_conn__{db_path}"
    conn = getattr(_thread_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        setattr(_thread_local, key, conn)
    return conn

def _canonical_key(s: str, leading_articles=DEFAULT_LEADING_ARTICLES) -> str:
    """
    Aggressive canonical form used to dedupe subject variants:
    - Unicode NFKC, lowercased, collapse whitespace
    - strip leading articles ("the", "a", "an")
    - remove all non-alphanumerics
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = _ws.sub(" ", s)
    for art in leading_articles:
        if s.startswith(art + " "):
            s = s[len(art) + 1:]
            break
    return _nonword.sub("", s)

def _subject_norm(s: str) -> str:
    """
    Gentler normalization used for presentation:
    - NFKC, lower, collapse spaces
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return _ws.sub(" ", s)

# ----- bootstrap / indices -----

def ensure_processed_index(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_map (
            canon_key TEXT PRIMARY KEY,
            sample_original TEXT
        )
    """)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm   ON queue(subject_norm)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_canon  ON queue(subject_canon)")
    conn.commit()

# ----- API -----

EnqResult = Tuple[str, int, str]  # (subject original, kept_hop, outcome: inserted | hop_reduced | ignored)

def init_cache(conn_or_path):
    if isinstance(conn_or_path, str):
        conn = get_thread_queue_conn(conn_or_path)
    else:
        conn = conn_or_path
    ensure_processed_index(conn)

def enqueue_subjects_processed(
    db_path_or_conn,
    items: Iterable[Tuple[str, int]],
    leading_articles=DEFAULT_LEADING_ARTICLES
) -> List[EnqResult]:
    """
    Enqueue with **canonical dedupe**:
      - canonical key (subject_canon) ensures only *one* row ever exists per real-world subject
      - if a new variant arrives with a *lower hop*, we *lower* the hop of the existing row (keep status as-is)
      - if it’s a true duplicate with same-or-higher hop → outcome 'ignored'

    Returns: list of (subject, kept_hop, outcome)
    """
    conn = db_path_or_conn if not isinstance(db_path_or_conn, str) else get_thread_queue_conn(db_path_or_conn)
    ensure_processed_index(conn)

    results: List[EnqResult] = []
    cur = conn.cursor()

    with conn:
        for subject, hop in items:
            if not isinstance(subject, str) or not subject.strip():
                continue

            canon = _canonical_key(subject, leading_articles=leading_articles)
            s_norm = _subject_norm(subject)

            # keep one sample per canonical key (for visibility)
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

            # Read back the kept hop
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
