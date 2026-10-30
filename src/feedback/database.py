import sqlite3
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
DATABASE_PATH = FEEDBACK_DIR / "feedback.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite database connection."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_feedback_database() -> None:
    """Create the feedback table and its indexes if they do not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL
                    CHECK (rating IN ('up', 'down')),
                comment TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_query_id
            ON feedback(query_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_rating
            ON feedback(rating)
            """
        )


def insert_feedback(
    query_id: str,
    question: str,
    answer: str,
    rating: str,
    comment: str | None = None,
    created_at: str | None = None,
) -> int:
    """Insert feedback and return its generated ID."""
    if rating not in {"up", "down"}:
        raise ValueError("Rating must be either 'up' or 'down'.")

    if created_at is None:
        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback (
                query_id,
                question,
                answer,
                rating,
                comment,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (query_id, question, answer, rating, comment, created_at),
        )

        return int(cursor.lastrowid)


def get_feedback_by_query_id(query_id: str) -> list[dict[str, Any]]:
    """Return all feedback entries associated with a query ID."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM feedback
            WHERE query_id = ?
            ORDER BY created_at DESC
            """,
            (query_id,),
        ).fetchall()

        return [dict(row) for row in rows]
