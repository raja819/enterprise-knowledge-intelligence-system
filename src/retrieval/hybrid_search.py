
import logging
import re

from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.vector_store import (
    get_qdrant_client,
    search_similar_chunks,
)

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Basic case-insensitive tokenization for BM25."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_indexed_chunks() -> list[dict]:
    """Read indexed document payloads from the local Qdrant collection."""
    client = get_qdrant_client()
    chunks = []
    offset = None

    try:
        if not client.collection_exists(settings.qdrant_collection):
            return []

        while True:
            points, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                text = payload.get("text", "")

                if not isinstance(text, str) or not text.strip():
                    continue

                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            key: value
                            for key, value in payload.items()
                            if key != "text"
                        },
                    }
                )

            if next_offset is None:
                break

            offset = next_offset

        return chunks

    finally:
        client.close()


def chunk_key(chunk: dict) -> tuple:
    metadata = chunk["metadata"]

    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_id"),
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Combine ranked lists using reciprocal rank fusion."""
    scores = {}
    documents = {}

    for ranked_list in ranked_lists:
        seen_in_list = set()

        for rank, document in enumerate(ranked_list, start=1):
            key = chunk_key(document)

            if key in seen_in_list:
                continue

            seen_in_list.add(key)
            documents[key] = document
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    results = []

    for key in ranked_keys:
        document = dict(documents[key])
        document["rrf_score"] = scores[key]
        results.append(document)

    return results


def hybrid_search(
    query: str,
    limit: int | None = None,
) -> list[dict]:
    """Combine BM25 lexical retrieval with semantic vector retrieval."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    final_limit = (
        limit if limit is not None else settings.retrieval_top_k
    )

    if final_limit < 1:
        raise ValueError("Search limit must be at least 1.")

    chunks = get_indexed_chunks()

    if not chunks:
        logger.warning("No indexed chunks are available.")
        return []

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    valid_pairs = [
        (chunk, tokens)
        for chunk, tokens in zip(chunks, tokenized_corpus)
        if tokens
    ]

    if not valid_pairs:
        return []

    corpus_chunks = [pair[0] for pair in valid_pairs]
    tokenized_corpus = [pair[1] for pair in valid_pairs]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    lexical_order = sorted(
        range(len(corpus_chunks)),
        key=lambda index: float(bm25_scores[index]),
        reverse=True,
    )

    candidate_k = max(final_limit, settings.retrieval_top_k)

    lexical_results = [
        {
            **corpus_chunks[index],
            "bm25_score": float(bm25_scores[index]),
        }
        for index in lexical_order[:candidate_k]
        if bm25_scores[index] > 0
    ]

    semantic_results = search_similar_chunks(
        query,
        limit=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        [lexical_results, semantic_results]
    )

    logger.info(
        "Hybrid search completed: lexical=%s, semantic=%s, fused=%s",
        len(lexical_results),
        len(semantic_results),
        len(fused_results),
    )

    return fused_results[:final_limit]
