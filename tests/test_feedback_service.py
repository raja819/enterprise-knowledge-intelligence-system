import pytest

from src.feedback import service


@pytest.fixture
def mock_insert(monkeypatch):
    def fake_insert_feedback(**kwargs):
        return 101

    monkeypatch.setattr(
        service,
        "insert_feedback",
        fake_insert_feedback,
    )


def test_submit_positive_feedback(mock_insert):
    result = service.submit_feedback(
        query_id=" query-001 ",
        question=" What is RAG? ",
        answer=" RAG combines retrieval and generation. ",
        rating="UP",
        comment=" Helpful answer. ",
    )

    assert result["feedback_id"] == 101
    assert result["query_id"] == "query-001"
    assert result["rating"] == "up"


def test_submit_feedback_without_comment(mock_insert):
    result = service.submit_feedback(
        query_id="query-002",
        question="What is RAG?",
        answer="Retrieval augmented generation.",
        rating="down",
    )

    assert result["message"] == "Feedback submitted successfully."


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_id", "   ", "Query ID cannot be empty."),
        ("question", "   ", "Question cannot be empty."),
        ("answer", "   ", "Answer cannot be empty."),
        ("rating", "neutral", "Rating must be either 'up' or 'down'."),
    ],
)
def test_submit_feedback_rejects_invalid_values(
    mock_insert,
    field,
    value,
    message,
):
    payload = {
        "query_id": "query-003",
        "question": "Test question",
        "answer": "Test answer",
        "rating": "up",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        service.submit_feedback(**payload)


def test_submit_feedback_rejects_long_comment(mock_insert):
    with pytest.raises(ValueError, match="2000 characters"):
        service.submit_feedback(
            query_id="query-004",
            question="Test question",
            answer="Test answer",
            rating="up",
            comment="x" * 2001,
        )


def test_submit_feedback_normalizes_empty_comment(mock_insert, monkeypatch):
    captured = {}

    def fake_insert_feedback(**kwargs):
        captured.update(kwargs)
        return 102

    monkeypatch.setattr(service, "insert_feedback", fake_insert_feedback)

    service.submit_feedback(
        query_id="query-005",
        question="Test question",
        answer="Test answer",
        rating="down",
        comment="   ",
    )

    assert captured["comment"] is None
