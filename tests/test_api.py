from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_rejects_whitespace_only_question():
    response = client.post(
        "/query",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_query_rejects_question_over_2000_characters():
    response = client.post(
        "/query",
        json={"question": "a" * 2001},
    )

    assert response.status_code == 422


def test_query_returns_answer_and_sources(monkeypatch):
    # Mock retrieval so this test does not need to run the embedding model.
    monkeypatch.setattr(
        "api.main.hybrid_search",
        lambda question, limit: [
            {
                "text": "ACME Retail revenue in 2025 was ₹12.5 crore.",
                "metadata": {
                    "source": "sample_report.pdf",
                    "page": 1,
                },
            }
        ],
    )

    monkeypatch.setattr(
        "api.main.rerank_results",
        lambda question, documents, top_k: documents,
    )

    monkeypatch.setattr(
        "api.main.generate_answer",
        lambda question, documents: {
            "answer": (
                "ACME Retail revenue in 2025 was ₹12.5 crore. [S1]"
            ),
            "sources": [
                {
                    "citation": "S1",
                    "source": "sample_report.pdf",
                    "page": 1,
                }
            ],
        },
    )

    response = client.post(
        "/query",
        json={
            "question": "What was the revenue in 2025?"
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["query_id"]
    assert "₹12.5 crore" in body["answer"]
    assert body["has_citations"] is True
    assert body["citations_valid"] is True
    assert body["sources"][0]["source"] == "sample_report.pdf"
    assert body["sources"][0]["page"] == 1
    assert body["answer_quality"]["is_acceptable"] is True
    assert body["answer_quality"]["has_citations"] is True


def test_query_handles_no_retrieved_documents(monkeypatch):
    monkeypatch.setattr(
        "api.main.hybrid_search",
        lambda question, limit: [],
    )

    response = client.post(
        "/query",
        json={"question": "Question about an unknown document"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query_id"]
    assert body["sources"] == []
    assert body["has_citations"] is False
    assert "could not find supporting documents" in body["answer"].lower()
    assert body["answer_quality"]["is_acceptable"] is False
    assert body["answer_quality"]["has_sources"] is False