import argparse
import logging
import time
from pathlib import Path

from src.config import PROJECT_ROOT
from src.ingestion.audit_logger import write_ingestion_audit
from src.ingestion.chunker import split_pages_into_chunks
from src.ingestion.pdf_loader import load_pdf
from src.logging_config import setup_logging
from src.retrieval.vector_store import index_chunks

logger = logging.getLogger(__name__)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_pdf(pdf_path: Path) -> dict:
    """Extract, chunk, index, and audit one PDF document."""

    started = time.perf_counter()
    pages_extracted = 0
    chunks_created = 0
    indexed_count = 0

    logger.info("Starting ingestion: %s", pdf_path.name)

    try:
        pages = load_pdf(pdf_path)
        pages_extracted = len(pages)

        chunks = split_pages_into_chunks(pages)
        chunks_created = len(chunks)

        indexed_count = index_chunks(chunks)

    except Exception as exc:
        duration_seconds = round(time.perf_counter() - started, 4)

        audit_record = {
            "file": pdf_path.name,
            "status": "failure",
            "pages_extracted": pages_extracted,
            "chunks_created": chunks_created,
            "chunks_indexed": indexed_count,
            "duration_seconds": duration_seconds,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        try:
            write_ingestion_audit(audit_record)
        except OSError:
            logger.exception(
                "Audit logging also failed for %s",
                pdf_path.name,
            )

        logger.exception("Failed to ingest %s", pdf_path.name)
        raise

    duration_seconds = round(time.perf_counter() - started, 4)

    result = {
        "file": pdf_path.name,
        "status": "success",
        "pages_extracted": pages_extracted,
        "chunks_created": chunks_created,
        "chunks_indexed": indexed_count,
        "duration_seconds": duration_seconds,
    }

    try:
        write_ingestion_audit(result)
    except OSError:
        logger.exception(
            "Ingestion succeeded but its audit record "
            "could not be written: %s",
            pdf_path.name,
        )

    logger.info("Ingestion completed: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the EKIS vector store."
    )
    parser.add_argument(
        "--file",
        help="PDF filename inside data/raw. If omitted, process all PDFs.",
    )
    args = parser.parse_args()

    setup_logging()

    if args.file:
        pdf_path = (RAW_DATA_DIR / args.file).resolve()

        if pdf_path.parent != RAW_DATA_DIR.resolve():
            parser.error("Provide a filename directly inside data/raw.")

        if pdf_path.suffix.lower() != ".pdf":
            parser.error("Only PDF files are supported.")

        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        parser.error("No PDF files found in data/raw.")

    results = []

    for pdf_path in pdf_files:
        try:
            results.append(ingest_pdf(pdf_path))
        except Exception:
            # ingest_pdf already logs the exception and audit record.
            continue

    print("\nINGESTION SUMMARY")

    for result in results:
        print(result)

    if len(results) != len(pdf_files):
        raise SystemExit("One or more PDFs failed to ingest.")


if __name__ == "__main__":
    main()
    