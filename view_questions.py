import sqlite3
from contextlib import closing

from db import DB_PATH

print("Database location:", DB_PATH.resolve())
print("Database exists:", DB_PATH.exists())

if not DB_PATH.exists():
    raise SystemExit("No database yet. Run the app first.")

with closing(sqlite3.connect(DB_PATH)) as conn:
    rows = conn.execute("""
        SELECT id, session_id, question, asked_at
        FROM questions
        ORDER BY id DESC
        LIMIT 20
    """).fetchall()

if not rows:
    print("No questions saved yet.")

for question_id, session_id, question, asked_at in rows:
    print(f"\nID: {question_id}")
    print(f"Session: {session_id}")
    print(f"Asked at (UTC): {asked_at}")
    print(f"Question: {question}")
    ## adding a comment to see git 