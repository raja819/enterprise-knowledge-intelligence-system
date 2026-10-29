import logging
import pytest

from scripts import ingest_documents


def test_ingest_pdf_records_success(monkeypatch):
    audit_records = []

    monkeypatch.setattr(
        ingest_documents,
        "load_pdf",
        lambda path: [{"text": "Page one", "metadata": {"page": 1}}],
    )
    monkeypatch.setattr(
        ingest_documents,
        "split_pages_into_chunks",
        lambda pages: [{"text": "Chunk one", "metadata": {"page": 1}}],
    )
    monkeypatch.setattr(ingest_documents, "index_chunks", lambda chunks: 1)
    monkeypatch.setattr(
        ingest_documents,
        "write_ingestion_audit",
        lambda record: audit_records.append(record),
    )

    result = ingest_documents.ingest_pdf(ingest_documents.Path("sample.pdf"))

    assert result["file"] == "sample.pdf"
    assert result["status"] == "success"
    assert result["pages_extracted"] == 1
    assert result["chunks_created"] == 1
    assert result["chunks_indexed"] == 1
    assert result["duration_seconds"] >= 0
    assert len(audit_records) == 1
    assert audit_records[0]["status"] == "success"


def test_ingest_pdf_records_failure(monkeypatch):
    audit_records = []

    def fail_loading(path):
        raise ValueError("Invalid PDF")

    monkeypatch.setattr(ingest_documents, "load_pdf", fail_loading)
    monkeypatch.setattr(
        ingest_documents,
        "write_ingestion_audit",
        lambda record: audit_records.append(record),
    )

    with pytest.raises(ValueError, match="Invalid PDF"):
        ingest_documents.ingest_pdf(ingest_documents.Path("broken.pdf"))

    assert len(audit_records) == 1
    assert audit_records[0]["file"] == "broken.pdf"
    assert audit_records[0]["status"] == "failure"
    assert audit_records[0]["error_type"] == "ValueError"
    assert audit_records[0]["error"] == "Invalid PDF"
    assert audit_records[0]["duration_seconds"] >= 0


def test_audit_failure_does_not_hide_ingestion_error(monkeypatch, caplog):
    def fail_loading(path):
        raise ValueError("Invalid PDF")

    def fail_audit(record):
        raise OSError("Audit disk unavailable")

    monkeypatch.setattr(ingest_documents, "load_pdf", fail_loading)
    monkeypatch.setattr(
        ingest_documents,
        "write_ingestion_audit",
        fail_audit,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Invalid PDF"):
            ingest_documents.ingest_pdf(ingest_documents.Path("broken.pdf"))

    assert "Audit logging also failed" in caplog.text
    assert "Failed to ingest broken.pdf" in caplog.text