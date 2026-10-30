import pytest

from src.feedback.analytics import (
    get_daily_feedback_trend,
    get_feedback_summary,
    get_low_rated_feedback,
)
from src.feedback.database import initialize_feedback_database, insert_feedback


@pytest.fixture
def feedback_db(tmp_path, monkeypatch):
    import src.feedback.database as database

    feedback_dir = tmp_path / "feedback"
    database.FEEDBACK_DIR = feedback_dir
    database.DATABASE_PATH = feedback_dir / "test_feedback.db"

    initialize_feedback_database()

    yield


def test_feedback_summary_empty_database(feedback_db):
    summary = get_feedback_summary()

    assert summary == {
        "total_feedback": 0,
        "positive": 0,
        "negative": 0,
        "positive_percentage": 0.0,
    }


def test_feedback_summary_with_records(feedback_db):
    insert_feedback("q1", "Question 1", "Answer 1", "up",
                    created_at="2026-10-28T10:00:00+00:00")
    insert_feedback("q2", "Question 2", "Answer 2", "up",
                    created_at="2026-10-28T11:00:00+00:00")
    insert_feedback("q3", "Question 3", "Answer 3", "down",
                    created_at="2026-10-28T12:00:00+00:00")

    summary = get_feedback_summary()

    assert summary["total_feedback"] == 3
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["positive_percentage"] == 66.67


def test_daily_feedback_trend(feedback_db):
    insert_feedback("q1", "Question 1", "Answer 1", "up",
                    created_at="2026-10-28T10:00:00+00:00")
    insert_feedback("q2", "Question 2", "Answer 2", "down",
                    created_at="2026-10-28T11:00:00+00:00")
    insert_feedback("q3", "Question 3", "Answer 3", "up",
                    created_at="2026-10-29T10:00:00+00:00")

    trend = get_daily_feedback_trend()

    assert len(trend) == 2
    assert trend[0]["feedback_date"] == "2026-10-28"
    assert trend[0]["total"] == 2
    assert trend[0]["positive"] == 1
    assert trend[0]["negative"] == 1


def test_low_rated_feedback(feedback_db):
    insert_feedback(
        "q1", "Bad question", "Bad answer", "down",
        comment="Answer was not relevant.",
        created_at="2026-10-28T10:00:00+00:00",
    )
    insert_feedback("q2", "Good question", "Good answer", "up")

    low_rated = get_low_rated_feedback()

    assert len(low_rated) == 1
    assert low_rated[0]["query_id"] == "q1"
    assert low_rated[0]["comment"] == "Answer was not relevant."


def test_low_rated_feedback_limit_validation(feedback_db):
    with pytest.raises(ValueError, match="at least 1"):
        get_low_rated_feedback(limit=0)