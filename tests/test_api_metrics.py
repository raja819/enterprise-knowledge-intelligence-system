import json

from fastapi.testclient import TestClient
from api.main import app
from src.observability import query_metrics

client = TestClient(app)


def read_metrics(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def setup_metrics_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "query_metrics.jsonl"
    monkeypatch.setattr(query_metrics, "METRICS_FILE", path)
    return path


def test_successful_query_records_performance_metrics(
    tmp_path, monkeypatch
):
    metrics_file = setup_metrics_file(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "api.main.hybrid_search",
        lambda question, limit: [
            {
                "text": "ACME Retail revenue was 12.5 crore.",
                "metadata": {"source": "sample_report.pdf", "page": 1},
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
            "answer": "Revenue was 12.5 crore. [S1]",
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
        json={"question": "What was the revenue?"},
    )

    assert response.status_code == 200

    records = read_metrics(metrics_file)
    assert len(records) == 1

    record = records[0]
    assert record["status"] == "success"
    assert record["retrieved_documents"] == 1
    assert record["reranked_documents"] == 1
    assert record["total_duration_ms"] >= 0
    assert record["retrieval_duration_ms"] >= 0
    assert record["reranking_duration_ms"] >= 0
    assert record["generation_duration_ms"] >= 0
    assert record["error_type"] is None
    assert record["request_id"]


def test_no_documents_records_no_documents_status(
    tmp_path, monkeypatch
):
    metrics_file = setup_metrics_file(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "api.main.hybrid_search",
        lambda question, limit: [],
    )

    response = client.post(
        "/query",
        json={"question": "Unknown document question"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []

    records = read_metrics(metrics_file)
    assert len(records) == 1
    assert records[0]["status"] == "no_documents"
    assert records[0]["retrieved_documents"] == 0
    assert records[0]["reranked_documents"] == 0


def test_retrieval_error_records_error_status(
    tmp_path, monkeypatch
):
    metrics_file = setup_metrics_file(tmp_path, monkeypatch)

    def fail_retrieval(question, limit):
        raise RuntimeError("Simulated retrieval failure")

    monkeypatch.setattr("api.main.hybrid_search", fail_retrieval)

    response = client.post(
        "/query",
        json={"question": "Test retrieval failure"},
    )

    assert response.status_code == 500

    records = read_metrics(metrics_file)
    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["error_type"] == "RuntimeError"
    assert records[0]["retrieved_documents"] == 0
