import argparse
import logging
from pathlib import Path

from src.config import PROJECT_ROOT
from src.ingestion.chunker import split_pages_into_chunks
from src.ingestion.pdf_loader import load_pdf
from src.logging_config import setup_logging
from src.retrieval.vector_store import index_chunks

logger = logging.getLogger(__name__)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_pdf(pdf_path: Path) -> dict:
    """Extract, chunk, and index one PDF document."""

    logger.info("Starting ingestion: %s", pdf_path.name)

    pages = load_pdf(pdf_path)
    chunks = split_pages_into_chunks(pages)
    indexed_count = index_chunks(chunks)

    result = {
        "file": pdf_path.name,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed_count,
    }

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
        except Exception as exc:
            logger.exception(
                "Failed to ingest %s: %s",
                pdf_path.name,
                exc,
            )

    print("\nINGESTION SUMMARY")

    for result in results:
        print(result)

    if len(results) != len(pdf_files):
        raise SystemExit("One or more PDFs failed to ingest.")


if __name__ == "__main__":
    main()
