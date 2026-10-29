import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "ingestion_audit.jsonl"


def write_ingestion_audit(record: dict[str, Any]) -> None:
    """Append one JSON audit record for a document ingestion attempt."""

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }

    try:
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as audit_file:
            audit_file.write(
                json.dumps(audit_record, ensure_ascii=False, default=str)
                + "\n"
            )
    except OSError:
        logger.exception("Could not write ingestion audit record")
        raise