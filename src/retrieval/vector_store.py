
import logging
import uuid

from qdrant_client import QdrantClient, models

from src.config import PROJECT_ROOT, settings
from src.retrieval.embeddings import (
    EMBEDDING_DIMENSION,
    generate_embeddings,
)

logger = logging.getLogger(__name__)

STORAGE_PATH = PROJECT_ROOT / "storage" / "qdrant_db"


def get_qdrant_client() -> QdrantClient:
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    return QdrantClient(path=str(STORAGE_PATH))


def ensure_collection(client: QdrantClient) -> None:
    collection_name = settings.qdrant_collection

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors

        if not isinstance(vectors, models.VectorParams):
            raise ValueError(
                "This project requires a single-vector Qdrant collection."
            )

        if vectors.size != EMBEDDING_DIMENSION:
            raise ValueError(
                "Existing collection has the wrong vector dimension. "
                "Use a new collection name or recreate the development index."
            )

        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    logger.info("Created Qdrant collection: %s", collection_name)


def index_chunks(chunks: list[dict]) -> int:
    if not chunks:
        raise ValueError("Chunks list cannot be empty.")

    texts = [chunk["text"] for chunk in chunks]
    vectors = generate_embeddings(texts)

    points = []

    for chunk, vector in zip(chunks, vectors):
        metadata = chunk["metadata"]
        source = metadata["source"]
        page = metadata["page"]
        chunk_id = metadata["chunk_id"]

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{source}:{page}:{chunk_id}:{chunk['text']}",
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    **metadata,
                },
            )
        )

    client = get_qdrant_client()

    try:
        ensure_collection(client)

        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )

        logger.info("Indexed %s chunks", len(points))
        return len(points)
    finally:
        client.close()


def search_similar_chunks(
    query: str,
    limit: int | None = None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    top_k = limit if limit is not None else settings.retrieval_top_k

    if top_k < 1:
        raise ValueError("Search limit must be at least 1.")

    query_vector = generate_embeddings([query])[0]
    client = get_qdrant_client()

    try:
        ensure_collection(client)

        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                    "score": point.score,
                }
            )

        return results
    finally:
        client.close()
