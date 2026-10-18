import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Load and cache the cross-encoder reranking model."""
    logger.info(
        "Loading reranker model: %s",
        RERANKER_MODEL_NAME,
    )

    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_results(
    query: str,
    documents: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Rerank retrieved documents using query-document relevance.

    Each document must contain:
        - text: retrieved chunk text
        - metadata: source and page information
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(documents, list):
        raise ValueError("Documents must be a list.")

    if not documents:
        return []

    final_top_k = (
        top_k if top_k is not None else settings.rerank_top_k
    )

    if final_top_k < 1:
        raise ValueError("top_k must be at least 1.")

    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Every document must be a dictionary.")

        if not isinstance(document.get("text"), str):
            raise ValueError("Every document must contain text.")

        if not document["text"].strip():
            raise ValueError("Document text cannot be empty.")

    model = get_reranker_model()

    query_document_pairs = [
        (query.strip(), document["text"])
        for document in documents
    ]

    scores = model.predict(
        query_document_pairs,
        show_progress_bar=False,
    )

    reranked_documents = []

    for document, score in zip(documents, scores):
        reranked_document = document.copy()
        reranked_document["rerank_score"] = float(score)
        reranked_documents.append(reranked_document)

    reranked_documents.sort(
        key=lambda document: document["rerank_score"],
        reverse=True,
    )

    logger.info(
        "Reranked %s documents; returning top %s",
        len(documents),
        min(final_top_k, len(reranked_documents)),
    )

    return reranked_documents[:final_top_k]