from pathlib import Path
import logging

import pymupdf

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_PDF_SIZE_MB = 50
MAX_PDF_PAGES = 2000


def load_pdf(file_path: str | Path) -> list[dict]:
    """Extract text page by page from a PDF inside data/raw."""
    path = Path(file_path).resolve()
    raw_dir = (PROJECT_ROOT / "data" / "raw").resolve()

    if not path.is_relative_to(raw_dir):
        raise ValueError("PDF must be inside data/raw.")

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.name}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF files are supported.")

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("PDF file is empty.")

    if file_size > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"PDF exceeds the {MAX_PDF_SIZE_MB} MB limit."
        )

    pages = []

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            if page_count == 0:
                raise ValueError("PDF contains no pages.")

            if page_count > MAX_PDF_PAGES:
                raise ValueError(
                    f"PDF exceeds the {MAX_PDF_PAGES}-page limit."
                )

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                if not text:
                    logger.warning(
                        "No extractable text on page %s of %s",
                        page_index + 1,
                        path.name,
                    )
                    continue

                pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": path.name,
                            "page": page_index + 1,
                            "total_pages": page_count,
                        },
                    }
                )

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Failed to read PDF: %s", path.name)
        raise ValueError(
            f"Could not process PDF: {path.name}"
        ) from exc

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned "
            "or contain only images."
        )

    logger.info(
        "Extracted text from %s of %s pages in %s",
        len(pages),
        pages[0]["metadata"]["total_pages"],
        path.name,
    )

    return pages