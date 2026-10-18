
import argparse

from src.config import PROJECT_ROOT
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import split_pages_into_chunks
from src.retrieval.vector_store import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, chunk and index a PDF."
    )
    parser.add_argument(
        "--file",
        default="sample_report.pdf",
        help="PDF filename inside data/raw/",
    )
    args = parser.parse_args()

    pdf_path = (
        PROJECT_ROOT / "data" / "raw" / args.file
    ).resolve()

    raw_dir = (PROJECT_ROOT / "data" / "raw").resolve()

    if not pdf_path.is_relative_to(raw_dir):
        raise ValueError("PDF must be inside data/raw/.")

    print(f"Processing: {pdf_path.name}")

    pages = load_pdf(pdf_path)
    print(f"Extracted pages: {len(pages)}")

    chunks = split_pages_into_chunks(pages)
    print(f"Created chunks: {len(chunks)}")

    indexed_count = index_chunks(chunks)
    print(f"Successfully indexed chunks: {indexed_count}")


if __name__ == "__main__":
    main()
