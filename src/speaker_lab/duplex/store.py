"""Single-run durable job results and append-only review records."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .protocol import content_hash


class RunStore:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    clip_id TEXT, condition TEXT, status TEXT NOT NULL, attempt INTEGER NOT NULL,
                    result TEXT, error TEXT, PRIMARY KEY(clip_id, condition));
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, record TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def start_attempt(self, clip_id: str, condition: str) -> int:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE clip_id=? AND condition=?", (clip_id, condition)).fetchone()
            if row and row["status"] in {"succeeded", "running"}:
                raise ValueError(f"Job is {row['status']}; do not restart a committed or active job")
            attempt = row["attempt"] + 1 if row else 1
            db.execute("INSERT INTO jobs VALUES(?,?, 'running', ?,NULL,NULL) "
                       "ON CONFLICT(clip_id,condition) DO UPDATE SET status='running',attempt=excluded.attempt,result=NULL,error=NULL",
                       (clip_id, condition, attempt))
            return attempt

    def finish_attempt(self, clip_id: str, condition: str, attempt: int, result: dict) -> None:
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self.connection() as db:
            changed = db.execute("UPDATE jobs SET status='succeeded',result=?,error=NULL "
                                 "WHERE clip_id=? AND condition=? AND attempt=? AND status='running'",
                                 (encoded, clip_id, condition, attempt)).rowcount
            if changed != 1:
                raise ValueError("Stale or non-running attempt cannot commit")

    def fail_attempt(self, clip_id: str, condition: str, attempt: int, error: str) -> None:
        with self.connection() as db:
            changed = db.execute("UPDATE jobs SET status='failed',error=? "
                                 "WHERE clip_id=? AND condition=? AND attempt=? AND status='running'",
                                 (error, clip_id, condition, attempt)).rowcount
            if changed != 1:
                raise ValueError("Stale or non-running attempt cannot fail")

    def iter_results(self) -> list[dict]:
        with self.connection() as db:
            return [json.loads(row["result"]) for row in db.execute(
                "SELECT result FROM jobs WHERE status='succeeded' ORDER BY clip_id,condition")]

    def iter_jobs(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT clip_id,condition,status,attempt,error FROM jobs ORDER BY clip_id,condition")]

    def put_review(self, review: dict) -> None:
        encoded = json.dumps(review, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO reviews VALUES(?,?)", (content_hash(review), encoded))

    def iter_reviews(self) -> list[dict]:
        with self.connection() as db:
            return [json.loads(row["record"]) for row in db.execute("SELECT record FROM reviews ORDER BY rowid")]
