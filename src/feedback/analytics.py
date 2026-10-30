from typing import Any

from src.feedback.database import get_connection


def get_feedback_summary() -> dict[str, Any]:
    """Return overall feedback statistics."""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_feedback,
                SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS negative
            FROM feedback
            """
        ).fetchone()

    total = row["total_feedback"] or 0
    positive = row["positive"] or 0
    negative = row["negative"] or 0

    positive_percentage = (
        round((positive / total) * 100, 2)
        if total > 0
        else 0.0
    )

    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "positive_percentage": positive_percentage,
    }


def get_daily_feedback_trend() -> list[dict[str, Any]]:
    """Return daily positive and negative feedback counts."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                substr(created_at, 1, 10) AS feedback_date,
                COUNT(*) AS total,
                SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS negative
            FROM feedback
            GROUP BY substr(created_at, 1, 10)
            ORDER BY feedback_date ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_low_rated_feedback(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent negative feedback for investigation."""

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                feedback_id,
                query_id,
                question,
                answer,
                rating,
                comment,
                created_at
            FROM feedback
            WHERE rating = 'down'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]