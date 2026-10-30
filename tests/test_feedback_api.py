from fastapi.testclient import TestClient

from api.main import app
from src.feedback import database

client = TestClient(app)


def setup_feedback_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        tmp_path / "feedback.db",
    )
    database.initialize_feedback_database()


def test_feedback_endpoint_saves_positive_feedback(
    tmp_path, monkeypatch
):
    setup_feedback_database(tmp_path, monkeypatch)

    response = client.post(
        "/feedback",
        json={
            "query_id": "query-101",
            "question": "What is RAG?",
            "answer": "RAG combines retrieval and generation.",
            "rating": "up",
            "comment": "Helpful answer.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["feedback_id"] == 1
    assert body["query_id"] == "query-101"
    assert body["rating"] == "up"
    assert body["message"] == "Feedback submitted successfully."

    saved_feedback = database.get_feedback_by_query_id("query-101")

    assert len(saved_feedback) == 1
    assert saved_feedback[0]["comment"] == "Helpful answer."


def test_feedback_endpoint_saves_negative_feedback(
    tmp_path, monkeypatch
):
    setup_feedback_database(tmp_path, monkeypatch)

    response = client.post(
        "/feedback",
        json={
            "query_id": "query-102",
            "question": "Explain vector search.",
            "answer": "Vector search uses embeddings.",
            "rating": "down",
        },
    )

    assert response.status_code == 200

    saved_feedback = database.get_feedback_by_query_id("query-102")

    assert saved_feedback[0]["rating"] == "down"
    assert saved_feedback[0]["comment"] is None


def test_feedback_endpoint_rejects_invalid_rating():
    response = client.post(
        "/feedback",
        json={
            "query_id": "query-103",
            "question": "Test question",
            "answer": "Test answer",
            "rating": "neutral",
        },
    )

    assert response.status_code == 422


def test_feedback_endpoint_rejects_empty_query_id():
    response = client.post(
        "/feedback",
        json={
            "query_id": "   ",
            "question": "Test question",
            "answer": "Test answer",
            "rating": "up",
        },
    )

    assert response.status_code == 422


def test_feedback_endpoint_rejects_long_comment():
    response = client.post(
        "/feedback",
        json={
            "query_id": "query-104",
            "question": "Test question",
            "answer": "Test answer",
            "rating": "up",
            "comment": "x" * 2001,
        },
    )

    assert response.status_code == 422
