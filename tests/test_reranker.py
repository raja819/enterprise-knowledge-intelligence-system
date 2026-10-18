import pytest

from src.retrieval.reranker import rerank_results


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank_results("   ", [])


def test_empty_documents_return_empty_list():
    assert rerank_results(
        "What was the revenue?",
        [],
    ) == []


def test_invalid_top_k_is_rejected():
    documents = [
        {
            "text": "Revenue was ₹12.5 crore.",
            "metadata": {
                "source": "sample_report.pdf",
                "page": 1,
            },
        }
    ]

    with pytest.raises(ValueError, match="at least 1"):
        rerank_results(
            "What was the revenue?",
            documents,
            top_k=0,
        )


def test_documents_must_contain_text():
    documents = [
        {
            "metadata": {
                "source": "sample_report.pdf",
                "page": 1,
            }
        }
    ]

    with pytest.raises(ValueError, match="contain text"):
        rerank_results(
            "What was the revenue?",
            documents,
        )