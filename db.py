# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("questions.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                asked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_question(session_id, question):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO questions (session_id, question) VALUES (?, ?)",
            (session_id, question),
        )
        conn.commit()
    finally:
        conn.close()