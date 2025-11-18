# db_models_articles.py
from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional

def open_articles_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            subject TEXT NOT NULL,
            hop INTEGER NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            infobox_json TEXT,
            sections_json TEXT NOT NULL,
            references_json TEXT,
            categories_json TEXT,
            model_name TEXT,
            strategy TEXT,
            overall_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(subject, hop)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS article_sink (
            subject TEXT NOT NULL,
            hop INTEGER NOT NULL,
            model_name TEXT,
            strategy TEXT,
            reason TEXT,
            raw TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def write_article(
    conn: sqlite3.Connection,
    subject: str, hop: int,
    title: str, summary: str,
    infobox_json: Optional[str], sections_json: str,
    references_json: Optional[str], categories_json: Optional[str],
    model_name: Optional[str], strategy: Optional[str],
    overall_conf: Optional[float]
) -> None:
    with conn:
        conn.execute("""
        INSERT OR REPLACE INTO articles
        (subject, hop, title, summary, infobox_json, sections_json, references_json, categories_json, model_name, strategy, overall_confidence)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (subject, hop, title, summary, infobox_json, sections_json, references_json, categories_json, model_name, strategy, overall_conf))

def write_article_sink(conn: sqlite3.Connection, subject: str, hop: int, model: str, strategy: str, reason: str, raw: str):
    with conn:
        conn.execute("""
        INSERT INTO article_sink (subject, hop, model_name, strategy, reason, raw)
        VALUES (?,?,?,?,?,?)
        """, (subject, hop, model, strategy, reason, raw))
