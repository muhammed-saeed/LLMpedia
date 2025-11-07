# # db_models.py
# from __future__ import annotations
# import sqlite3
# import unicodedata, re
# from typing import Iterable, Tuple, Literal

# from settings import QUEUE_DDL, FACTS_DDL

# _WS = re.compile(r"\s+")

# def normalize_subject(s: str) -> str:
#     if not isinstance(s, str):
#         return ""
#     s = unicodedata.normalize("NFKC", s)
#     s = _WS.sub(" ", s.strip())
#     return s.lower()

# def _open_sqlite(path: str) -> sqlite3.Connection:
#     conn = sqlite3.connect(path, check_same_thread=False)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA synchronous=NORMAL;")
#     conn.execute("PRAGMA temp_store=MEMORY;")
#     conn.execute("PRAGMA busy_timeout=5000;")
#     # mmap_size may fail on some platforms; you can keep or drop:
#     try:
#         conn.execute("PRAGMA mmap_size=30000000000;")
#     except sqlite3.OperationalError:
#         pass
#     conn.commit()
#     return conn

# def _ensure_queue_indexes(conn: sqlite3.Connection):
#     cur = conn.cursor()
#     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_hop ON queue(subject, hop)")
#     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm ON queue(subject_norm)")
#     conn.commit()

# def _ensure_facts_indexes(conn: sqlite3.Connection):
#     cur = conn.cursor()
#     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_triples ON triples_accepted(subject, predicate, object, hop)")
#     conn.commit()

# def open_queue_db(path: str) -> sqlite3.Connection:
#     conn = _open_sqlite(path)
#     conn.executescript(QUEUE_DDL)
#     _ensure_queue_indexes(conn)
#     return conn

# def open_facts_db(path: str) -> sqlite3.Connection:
#     conn = _open_sqlite(path)
#     conn.executescript(FACTS_DDL)
#     _ensure_facts_indexes(conn)
#     return conn

# EnqResult = Tuple[str, int, Literal["inserted", "hop_reduced", "ignored"]]

# def enqueue_subjects(db: sqlite3.Connection, items: Iterable[Tuple[str, int]]) -> list[EnqResult]:
#     """
#     (Kept for backward-compat; not used by the processed queue.)
#     """
#     cur = db.cursor()
#     results: list[EnqResult] = []

#     for subject, hop in items:
#         subj_norm = normalize_subject(subject)

#         cur.execute("SELECT hop FROM queue WHERE subject_norm=?", (subj_norm,))
#         row = cur.fetchone()
#         before_hop = row[0] if row else None

#         cur.execute(
#             """
#             INSERT INTO queue(subject, subject_norm, hop, status, retries)
#             VALUES (?, ?, ?, 'pending', 0)
#             ON CONFLICT(subject_norm) DO UPDATE SET
#               hop = CASE WHEN excluded.hop < hop THEN excluded.hop ELSE hop END
#             """,
#             (subject, subj_norm, hop),
#         )

#         cur.execute("SELECT hop FROM queue WHERE subject_norm=?", (subj_norm,))
#         kept_hop = cur.fetchone()[0]

#         if before_hop is None:
#             results.append((subject, kept_hop, "inserted"))
#         elif kept_hop < before_hop:
#             results.append((subject, kept_hop, "hop_reduced"))
#         else:
#             results.append((subject, kept_hop, "ignored"))

#     db.commit()
#     return results

# def reset_working_to_pending(conn: sqlite3.Connection) -> int:
#     cur = conn.cursor()
#     cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
#     conn.commit()
#     return cur.rowcount

# def queue_has_rows(conn: sqlite3.Connection) -> bool:
#     cur = conn.cursor()
#     cur.execute("SELECT 1 FROM queue LIMIT 1")
#     return cur.fetchone() is not None

# def count_queue(conn: sqlite3.Connection):
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
#     return done, working, pending, done + working + pending

# # -----------------------
# # Hardened triple writes
# # -----------------------

# def _sanitize_row(row):
#     # row: (subject, predicate, object, hop, model_name, strategy, confidence)
#     s, p, o, h, m, st, c = row

#     def as_str(x):
#         if x is None:
#             return ""
#         if isinstance(x, str):
#             return x
#         return str(x)

#     s = as_str(s)
#     p = as_str(p)
#     o = as_str(o)
#     m = as_str(m)
#     st = as_str(st)

#     try:
#         h = int(h)
#     except Exception:
#         h = 0

#     try:
#         c = float(c) if c is not None else None
#     except Exception:
#         c = None

#     return (s, p, o, h, m, st, c)

# def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
#     rows = [ _sanitize_row(r) for r in rows if r ]
#     if not rows:
#         return
#     cur = db.cursor()
#     cur.executemany(
#         """INSERT OR IGNORE INTO triples_accepted
#            (subject, predicate, object, hop, model_name, strategy, confidence)
#            VALUES (?, ?, ?, ?, ?, ?, ?)""",
#         rows,
#     )
#     db.commit()

# def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
#     if not rows:
#         return
#     # sanitize + pad reason
#     clean_rows = []
#     for r in rows:
#         s, p, o, h, m, st, c, reason = r
#         s, p, o, h, m, st, c = _sanitize_row((s, p, o, h, m, st, c))
#         reason = "" if reason is None else (reason if isinstance(reason, str) else str(reason))
#         clean_rows.append((s, p, o, h, m, st, c, reason))

#     cur = db.cursor()
#     cur.executemany(
#         """INSERT INTO triples_sink
#            (subject, predicate, object, hop, model_name, strategy, confidence, reason)
#            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
#         clean_rows,
#     )
#     db.commit()
# db_models.py
from __future__ import annotations
import sqlite3
import unicodedata, re
from typing import Iterable, Tuple, Literal

from settings import QUEUE_DDL, FACTS_DDL

_WS = re.compile(r"\s+")

def normalize_subject(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub(" ", s.strip())
    return s.lower()

def _open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        conn.execute("PRAGMA mmap_size=30000000000;")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return conn

def _ensure_queue_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm  ON queue(subject_norm)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_canon ON queue(subject_canon)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_queue_status_hop ON queue(status, hop)")
    conn.commit()

def _ensure_facts_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_triples ON triples_accepted(subject, predicate, object, hop)")
    conn.commit()

def open_queue_db(path: str) -> sqlite3.Connection:
    conn = _open_sqlite(path)
    conn.executescript(QUEUE_DDL)
    _ensure_queue_indexes(conn)
    return conn

def open_facts_db(path: str) -> sqlite3.Connection:
    conn = _open_sqlite(path)
    conn.executescript(FACTS_DDL)
    _ensure_facts_indexes(conn)
    return conn

EnqResult = Tuple[str, int, Literal["inserted", "hop_reduced", "ignored"]]

def reset_working_to_pending(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
    conn.commit()
    return cur.rowcount

def queue_has_rows(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM queue LIMIT 1")
    return cur.fetchone() is not None

def count_queue(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
    return done, working, pending, done + working + pending

# -------- triple writers --------

def _sanitize_row(row):
    s, p, o, h, m, st, c = row
    def as_str(x):
        if x is None: return ""
        return x if isinstance(x, str) else str(x)
    s, p, o, m, st = as_str(s), as_str(p), as_str(o), as_str(m), as_str(st)
    try: h = int(h)
    except Exception: h = 0
    try: c = float(c) if c is not None else None
    except Exception: c = None
    return (s, p, o, h, m, st, c)

def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
    rows = [ _sanitize_row(r) for r in rows if r ]
    if not rows:
        return
    cur = db.cursor()
    cur.executemany(
        """INSERT OR IGNORE INTO triples_accepted
           (subject, predicate, object, hop, model_name, strategy, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()

def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
    if not rows:
        return
    clean_rows = []
    for r in rows:
        s, p, o, h, m, st, c, reason = r
        s, p, o, h, m, st, c = _sanitize_row((s, p, o, h, m, st, c))
        reason = "" if reason is None else (reason if isinstance(reason, str) else str(reason))
        clean_rows.append((s, p, o, h, m, st, c, reason))

    cur = db.cursor()
    cur.executemany(
        """INSERT INTO triples_sink
           (subject, predicate, object, hop, model_name, strategy, confidence, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        clean_rows,
    )
    db.commit()
