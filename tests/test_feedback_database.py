import sqlite3
import pytest

from src.feedback import database


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    """Use a temporary SQLite database for each test."""
    test_database = tmp_path / "feedback.db"

    monkeypatch.setattr(database, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(database, "DATABASE_PATH", test_database)

    database.initialize_feedback_database()

    return test_database


def test_initialize_feedback_database_creates_table(temporary_database):
    with sqlite3.connect(temporary_database) as connection:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'feedback'
            """
        ).fetchone()

    assert result is not None


def test_insert_feedback_returns_id(temporary_database):
    feedback_id = database.insert_feedback(
        query_id="query-001",
        question="What was the revenue?",
        answer="Revenue was ₹12.5 crore.",
        rating="up",
        comment="Helpful answer.",
    )

    assert feedback_id == 1


def test_get_feedback_by_query_id(temporary_database):
    database.insert_feedback(
        query_id="query-002",
        question="What was the revenue?",
        answer="Revenue was ₹12.5 crore.",
        rating="up",
        comment="Accurate.",
    )

    results = database.get_feedback_by_query_id("query-002")

    assert len(results) == 1
    assert results[0]["query_id"] == "query-002"
    assert results[0]["rating"] == "up"
    assert results[0]["comment"] == "Accurate."


def test_feedback_is_linked_to_correct_query(temporary_database):
    database.insert_feedback(
        query_id="query-101",
        question="Question one",
        answer="Answer one",
        rating="up",
    )

    database.insert_feedback(
        query_id="query-202",
        question="Question two",
        answer="Answer two",
        rating="down",
    )

    results = database.get_feedback_by_query_id("query-101")

    assert len(results) == 1
    assert results[0]["question"] == "Question one"


def test_invalid_rating_is_rejected(temporary_database):
    with pytest.raises(ValueError, match="Rating must be"):
        database.insert_feedback(
            query_id="query-003",
            question="Test question",
            answer="Test answer",
            rating="neutral",
        )
