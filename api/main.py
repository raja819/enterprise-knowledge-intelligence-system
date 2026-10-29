import logging
import time

from fastapi import FastAPI, HTTPException
from api.schemas import QueryRequest, QueryResponse
from src.config import settings
from src.evaluation.answer_quality import evaluate_answer_quality
from src.generation.answer_generator import generate_answer
from src.generation.citation_validator import validate_citations
from src.logging_config import setup_logging
from src.observability.query_metrics import record_query_metric
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Intelligence System",
    description=(
        "A document-grounded RAG API with hybrid retrieval, "
        "reranking, and local LLM answer generation."
    ),
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
    }


@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(
    request: QueryRequest,
) -> QueryResponse:
    total_start = time.perf_counter()
    retrieval_duration_ms = 0.0
    reranking_duration_ms = 0.0
    generation_duration_ms = 0.0
    retrieved_documents = 0
    reranked_documents = 0
    query_status = "success"
    error_type = None

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=422,
                detail="Question cannot be empty.",
            )

        retrieval_start = time.perf_counter()
        try:
            candidates = hybrid_search(
                question,
                limit=settings.retrieval_top_k,
            )
        finally:
            retrieval_duration_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

        retrieved_documents = len(candidates)

        # Case 1: No relevant documents were retrieved.
        if not candidates:
            query_status = "no_documents"

            return QueryResponse(
                answer=(
                    "I could not find supporting documents "
                    "to answer this question."
                ),
                sources=[],
                citations_valid=True,
                has_citations=False,
                missing_citations=False,
                answer_quality={
                    "is_acceptable": False,
                    "has_answer": True,
                    "has_citations": False,
                    "has_sources": False,
                    "is_abstention": True,
                    "issues": ["no_sources"],
                },
            )

        # Case 2: Rerank retrieved documents.
        reranking_start = time.perf_counter()
        try:
            ranked_documents = rerank_results(
                question,
                candidates,
                top_k=settings.rerank_top_k,
            )
        finally:
            reranking_duration_ms = (
                time.perf_counter() - reranking_start
            ) * 1000

        reranked_documents = len(ranked_documents)

        # Case 3: Generate an answer using the local LLM.
        generation_start = time.perf_counter()
        try:
            result = generate_answer(
                question,
                ranked_documents,
            )
        finally:
            generation_duration_ms = (
                time.perf_counter() - generation_start
            ) * 1000

        # Case 4: Validate citations and answer quality.
        validation = validate_citations(
            result["answer"],
            result["sources"],
        )

        quality = evaluate_answer_quality(
            result["answer"],
            result["sources"],
        )

        logger.info(
            (
                "Query completed: retrieved=%s, reranked=%s, "
                "citations_valid=%s, missing_citations=%s, "
                "answer_acceptable=%s"
            ),
            retrieved_documents,
            reranked_documents,
            validation["is_valid"],
            validation["missing_citations"],
            quality["is_acceptable"],
        )

        # Case 5: Return the generated answer and validation results.
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            citations_valid=validation["is_valid"],
            has_citations=validation["has_citations"],
            missing_citations=validation["missing_citations"],
            answer_quality=quality,
        )

    except HTTPException as exc:
        query_status = "error"
        error_type = type(exc).__name__
        raise

    except Exception as exc:
        query_status = "error"
        error_type = type(exc).__name__

        logger.exception("Knowledge-base query failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "The query could not be processed. "
                "Check the application logs for details."
            ),
        ) from exc

    finally:
        total_duration_ms = (
            time.perf_counter() - total_start
        ) * 1000

        try:
            record_query_metric(
                total_duration_ms=total_duration_ms,
                retrieval_duration_ms=retrieval_duration_ms,
                reranking_duration_ms=reranking_duration_ms,
                generation_duration_ms=generation_duration_ms,
                retrieved_documents=retrieved_documents,
                reranked_documents=reranked_documents,
                status=query_status,
                error_type=error_type,
            )
        except Exception:
            # Observability must not replace the query's actual result.
            logger.exception("Query performance monitoring failed")
