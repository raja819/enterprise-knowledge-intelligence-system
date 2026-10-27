import pytest
import src.retrieval.hybrid_search as hybrid_search_module

from src.retrieval.hybrid_search import (
    reciprocal_rank_fusion,
    tokenize,
    hybrid_search,
)


def test_tokenize_normalizes_case():
    assert tokenize("Annual Revenue 2025") == [
        "annual",
        "revenue",
        "2025",
    ]


def test_rrf_prioritizes_documents_in_both_lists():
    document_a = {
        "text": "Revenue was ₹12.5 crore",
        "metadata": {
            "source": "sample_report.pdf",
            "page": 1,
            "chunk_id": 0,
        },
    }

    document_b = {
        "text": "Customer satisfaction was 87%",
        "metadata": {
            "source": "sample_report.pdf",
            "page": 2,
            "chunk_id": 1,
        },
    }

    results = reciprocal_rank_fusion(
        [
            [document_a, document_b],
            [document_b, document_a],
        ]
    )

    assert len(results) == 2
    assert results[0]["rrf_score"] == pytest.approx(
        results[1]["rrf_score"]
    )


def test_hybrid_search_rejects_empty_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid_search("   ")


def test_hybrid_search_returns_relevant_document(monkeypatch):
    sample_chunk = {
        "text": "The annual revenue in 2025 was ₹12.5 crore.",
        "metadata": {
            "source": "sample_report.pdf",
            "page": 1,
            "chunk_id": 0,
        },
    }

    # Use predictable test data instead of the local Qdrant database.
    monkeypatch.setattr(
        hybrid_search_module,
        "get_indexed_chunks",
        lambda: [sample_chunk],
    )

    monkeypatch.setattr(
        hybrid_search_module,
        "search_similar_chunks",
        lambda query, limit: [sample_chunk],
    )

    results = hybrid_search(
        "What was the annual revenue in 2025?",
        limit=3,
    )

    assert len(results) > 0

    assert any(
        "12.5 crore" in result["text"].lower()
        for result in results
    )

    for result in results:
        assert result["metadata"]["source"]
        assert result["metadata"]["page"] >= 1
        assert "rrf_score" in result


def test_rrf_preserves_bm25_and_vector_scores():
    bm25_document = {
        "text": "Revenue increased.",
        "metadata": {
            "source": "sample_report.pdf",
            "page": 1,
            "chunk_id": 0,
        },
        "bm25_score": 1.14,
    }

    vector_document = {
        "text": "Revenue increased.",
        "metadata": {
            "source": "sample_report.pdf",
            "page": 1,
            "chunk_id": 0,
        },
        "score": 0.76,
        "vector_score": 0.76,
    }

    results = reciprocal_rank_fusion(
        [[bm25_document], [vector_document]]
    )

    assert len(results) == 1
    assert results[0]["bm25_score"] == 1.14
    assert results[0]["vector_score"] == 0.76
    assert results[0]["rrf_score"] > 0
