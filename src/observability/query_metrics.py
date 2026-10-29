import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

METRICS_FILE = PROJECT_ROOT / "logs" / "query_metrics.jsonl"
_FILE_LOCK = threading.Lock()


def record_query_metric(
    *,
    total_duration_ms: float,
    retrieval_duration_ms: float,
    reranking_duration_ms: float,
    generation_duration_ms: float,
    retrieved_documents: int,
    reranked_documents: int,
    status: str,
    error_type: str | None = None,
) -> dict[str, Any]:
    """Append one query-performance record to a JSONL file."""

    durations = {
        "total_duration_ms": total_duration_ms,
        "retrieval_duration_ms": retrieval_duration_ms,
        "reranking_duration_ms": reranking_duration_ms,
        "generation_duration_ms": generation_duration_ms,
    }

    for name, value in durations.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric.")
        if value < 0 or not float(value) < float("inf"):
            raise ValueError(f"{name} must be finite and non-negative.")

    for name, value in (
        ("retrieved_documents", retrieved_documents),
        ("reranked_documents", reranked_documents),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    if status not in {"success", "error", "no_documents"}:
        raise ValueError("Unsupported query status.")

    if error_type is not None and not isinstance(error_type, str):
        raise ValueError("error_type must be a string or None.")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid4()),
        **{name: round(float(value), 3) for name, value in durations.items()},
        "retrieved_documents": retrieved_documents,
        "reranked_documents": reranked_documents,
        "status": status,
        "error_type": error_type,
    }

    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with _FILE_LOCK:
            with METRICS_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    except OSError:
        # Monitoring failures should be visible but must not expose
        # internal details or interrupt an otherwise successful query.
        logger.exception("Could not persist query performance metrics")

    return record
