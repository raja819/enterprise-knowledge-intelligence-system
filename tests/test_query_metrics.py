import json
import pytest

from src.observability import query_metrics


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "query_metrics.jsonl"
    monkeypatch.setattr(query_metrics, "METRICS_FILE", path)
    return path


def sample_metrics(**overrides):
    values = {
        "total_duration_ms": 125.4567,
        "retrieval_duration_ms": 25.1,
        "reranking_duration_ms": 40.2,
        "generation_duration_ms": 55.3,
        "retrieved_documents": 10,
        "reranked_documents": 5,
        "status": "success",
    }
    values.update(overrides)
    return values


def test_record_query_metric_writes_jsonl_record(metrics_file):
    record = query_metrics.record_query_metric(**sample_metrics())

    assert metrics_file.exists()
    lines = metrics_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    saved = json.loads(lines[0])
    assert saved["request_id"] == record["request_id"]
    assert saved["status"] == "success"
    assert saved["total_duration_ms"] == 125.457
    assert saved["retrieved_documents"] == 10
    assert saved["reranked_documents"] == 5
    assert saved["error_type"] is None
    assert "timestamp" in saved


def test_each_record_has_a_unique_request_id(metrics_file):
    first = query_metrics.record_query_metric(**sample_metrics())
    second = query_metrics.record_query_metric(**sample_metrics())

    assert first["request_id"] != second["request_id"]
    assert len(metrics_file.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    "field",
    [
        "total_duration_ms",
        "retrieval_duration_ms",
        "reranking_duration_ms",
        "generation_duration_ms",
    ],
)
@pytest.mark.parametrize("invalid_value", [-1, float("inf"), float("nan"), True])
def test_invalid_durations_are_rejected(field, invalid_value, metrics_file):
    with pytest.raises(ValueError):
        query_metrics.record_query_metric(
            **sample_metrics(**{field: invalid_value})
        )

    assert not metrics_file.exists()


@pytest.mark.parametrize(
    "field",
    ["retrieved_documents", "reranked_documents"],
)
@pytest.mark.parametrize("invalid_value", [-1, 1.5, True])
def test_invalid_document_counts_are_rejected(field, invalid_value, metrics_file):
    with pytest.raises(ValueError):
        query_metrics.record_query_metric(
            **sample_metrics(**{field: invalid_value})
        )

    assert not metrics_file.exists()


def test_unsupported_status_is_rejected(metrics_file):
    with pytest.raises(ValueError, match="Unsupported query status"):
        query_metrics.record_query_metric(
            **sample_metrics(status="unknown")
        )

    assert not metrics_file.exists()


def test_error_metadata_can_be_recorded(metrics_file):
    record = query_metrics.record_query_metric(
        **sample_metrics(status="error", error_type="RuntimeError")
    )

    saved = json.loads(metrics_file.read_text(encoding="utf-8").strip())
    assert record["status"] == "error"
    assert saved["error_type"] == "RuntimeError"


def test_monitoring_file_error_does_not_raise(metrics_file, monkeypatch):
    def fail_open(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        query_metrics.Path,
        "open",
        lambda *args, **kwargs: fail_open(),
    )

    record = query_metrics.record_query_metric(**sample_metrics())

    assert record["status"] == "success"
