
import pytest

from src.config import PROJECT_ROOT
from src.ingestion.pdf_loader import load_pdf


def test_missing_pdf_raises_error():
    missing_file = (
        PROJECT_ROOT / "data" / "raw" / "does_not_exist.pdf"
    )

    with pytest.raises(FileNotFoundError):
        load_pdf(missing_file)


def test_non_pdf_file_is_rejected(tmp_path):
    non_pdf = PROJECT_ROOT / "data" / "raw" / "not_a_pdf.txt"
    non_pdf.write_text("This is not a PDF.", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="Only PDF"):
            load_pdf(non_pdf)
    finally:
        non_pdf.unlink(missing_ok=True)


def test_extracted_pages_contain_metadata():
    pdf_path = PROJECT_ROOT / "data" / "raw" / "sample_report.pdf"

    if not pdf_path.exists():
        pytest.skip("Add sample_report.pdf to data/raw first.")

    pages = load_pdf(pdf_path)

    assert len(pages) > 0

    for page in pages:
        assert page["text"].strip()
        assert page["metadata"]["source"] == "sample_report.pdf"
        assert page["metadata"]["page"] >= 1
        assert page["metadata"]["total_pages"] >= page["metadata"]["page"]

def test_pdf_page_limit_is_enforced(monkeypatch):
    from src.ingestion import pdf_loader

    pdf_path = (
        pdf_loader.PROJECT_ROOT
        / "data"
        / "raw"
        / "sample_report.pdf"
    )

    class FakeDocument:
        is_encrypted = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __len__(self):
            return pdf_loader.MAX_PDF_PAGES + 1

    monkeypatch.setattr(
        pdf_loader.pymupdf,
        "open",
        lambda path: FakeDocument(),
    )

    try:
        pdf_loader.load_pdf(pdf_path)
        assert False, "Expected oversized page count to be rejected."
    except ValueError as exc:
        assert "page limit" in str(exc).lower()