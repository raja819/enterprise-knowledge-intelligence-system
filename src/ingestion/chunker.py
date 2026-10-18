
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


def split_pages_into_chunks(
    pages: list[dict],
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk retains its original source and page metadata.
    """
    if not pages:
        raise ValueError("Pages list cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        text = page.get("text", "").strip()
        metadata = page.get("metadata", {})

        if not text:
            continue

        source = metadata.get("source")
        page_number = metadata.get("page")

        if not source or not isinstance(page_number, int):
            raise ValueError(
                "Each page must contain source and integer page metadata."
            )

        documents = splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "source": source,
                    "page": page_number,
                }
            ],
        )

        for document in documents:
            chunk_text = document.page_content.strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **document.metadata,
                        "chunk_id": len(chunks),
                    },
                }
            )

    if not chunks:
        raise ValueError("No text chunks were created.")

    logger.info(
        "Created %s chunks from %s PDF pages",
        len(chunks),
        len(pages),
    )

    return chunks
