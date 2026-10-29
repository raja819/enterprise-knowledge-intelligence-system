
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


def test_empty_pdf_file_is_rejected(tmp_path):
    from src.ingestion.pdf_loader import load_pdf

    pdf_path = PROJECT_ROOT / "data" / "raw" / "empty_test.pdf"
    pdf_path.write_bytes(b"")

    try:
        with pytest.raises(ValueError, match="empty"):
            load_pdf(pdf_path)
    finally:
        pdf_path.unlink(missing_ok=True)


def test_oversized_pdf_file_is_rejected(monkeypatch):
    from src.ingestion import pdf_loader

    pdf_path = (
        PROJECT_ROOT / "data" / "raw" / "sample_report.pdf"
    )

    if not pdf_path.exists():
        pytest.skip("sample_report.pdf is required for this test.")

    monkeypatch.setattr(pdf_loader, "MAX_PDF_SIZE_MB", 0)

    with pytest.raises(ValueError, match="exceeds"):
        pdf_loader.load_pdf(pdf_path)


def test_file_outside_raw_directory_is_rejected(tmp_path):
    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="inside data/raw"):
        load_pdf(outside_file)


def test_corrupted_pdf_is_rejected(tmp_path, monkeypatch):
    from src.ingestion import pdf_loader

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(pdf_loader, "PROJECT_ROOT", tmp_path)

    pdf_path = raw_dir / "corrupted_test.pdf"
    pdf_path.write_bytes(b"This is not a valid PDF document.")

    with pytest.raises(ValueError, match="Could not process PDF"):
        pdf_loader.load_pdf(pdf_path)


def test_pdf_with_no_extractable_text_is_rejected(
    tmp_path,
    monkeypatch,
):
    from src.ingestion import pdf_loader

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(pdf_loader, "PROJECT_ROOT", tmp_path)

    pdf_path = raw_dir / "no_text.pdf"
    pdf_path.write_bytes(b"placeholder")

    class FakePage:
        def get_text(self, kind):
            return ""

    class FakeDocument:
        is_encrypted = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([FakePage()])

    monkeypatch.setattr(
        pdf_loader.pymupdf,
        "open",
        lambda path: FakeDocument(),
    )

    with pytest.raises(ValueError, match="No extractable text"):
        pdf_loader.load_pdf(pdf_path)


def test_encrypted_pdf_is_rejected(tmp_path, monkeypatch):
    from src.ingestion import pdf_loader

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(pdf_loader, "PROJECT_ROOT", tmp_path)

    pdf_path = raw_dir / "encrypted.pdf"
    pdf_path.write_bytes(b"placeholder")

    class FakeDocument:
        is_encrypted = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        pdf_loader.pymupdf,
        "open",
        lambda path: FakeDocument(),
    )

    with pytest.raises(ValueError, match="Password-protected"):
        pdf_loader.load_pdf(pdf_path)