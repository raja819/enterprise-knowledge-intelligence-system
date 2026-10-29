import json
import logging
import pytest

from src.ingestion import audit_logger


def test_audit_record_is_written_as_jsonl(tmp_path, monkeypatch):
    audit_path = tmp_path / "logs" / "ingestion_audit.jsonl"
    monkeypatch.setattr(audit_logger, "AUDIT_LOG_PATH", audit_path)

    record = {
        "file": "sample_report.pdf",
        "status": "success",
        "pages_extracted": 3,
        "chunks_created": 6,
        "chunks_indexed": 6,
        "duration_seconds": 1.25,
    }

    audit_logger.write_ingestion_audit(record)

    lines = audit_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    saved = json.loads(lines[0])
    assert saved["file"] == "sample_report.pdf"
    assert saved["status"] == "success"
    assert saved["pages_extracted"] == 3
    assert saved["chunks_created"] == 6
    assert saved["chunks_indexed"] == 6
    assert saved["duration_seconds"] == 1.25
    assert saved["timestamp"]


def test_audit_logger_appends_multiple_records(tmp_path, monkeypatch):
    audit_path = tmp_path / "ingestion_audit.jsonl"
    monkeypatch.setattr(audit_logger, "AUDIT_LOG_PATH", audit_path)

    audit_logger.write_ingestion_audit(
        {"file": "first.pdf", "status": "success"}
    )
    audit_logger.write_ingestion_audit(
        {"file": "second.pdf", "status": "failure"}
    )

    lines = audit_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["file"] == "first.pdf"
    assert json.loads(lines[1])["file"] == "second.pdf"


def test_audit_write_failure_is_logged_and_raised(
    tmp_path, monkeypatch, caplog
):
    audit_path = tmp_path / "audit.jsonl"
    audit_path.mkdir()

    monkeypatch.setattr(audit_logger, "AUDIT_LOG_PATH", audit_path)

    with caplog.at_level(logging.ERROR, logger=audit_logger.__name__):
        with pytest.raises(OSError):
            audit_logger.write_ingestion_audit(
                {"file": "sample_report.pdf", "status": "success"}
            )

    assert "Could not write ingestion audit record" in caplog.text