
import pytest

from src.ingestion.chunker import split_pages_into_chunks


def test_empty_pages_are_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        split_pages_into_chunks([])


def test_chunks_preserve_source_and_page():
    pages = [
        {
            "text": (
                "ACME Retail Company generated annual revenue. "
                "The company served many customers. "
            ) * 30,
            "metadata": {
                "source": "sample_report.pdf",
                "page": 2,
                "total_pages": 5,
            },
        }
    ]

    chunks = split_pages_into_chunks(pages)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["text"].strip()
        assert chunk["metadata"]["source"] == "sample_report.pdf"
        assert chunk["metadata"]["page"] == 2
        assert isinstance(chunk["metadata"]["chunk_id"], int)
        assert "start_index" in chunk["metadata"]


def test_missing_page_metadata_is_rejected():
    pages = [
        {
            "text": "This is a test document.",
            "metadata": {},
        }
    ]

    with pytest.raises(ValueError, match="source and integer"):
        split_pages_into_chunks(pages)
