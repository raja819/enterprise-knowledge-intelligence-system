import logging
import time

from fastapi import FastAPI, HTTPException

from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings
from src.evaluation.answer_quality import evaluate_answer_quality
from src.feedback.service import submit_feedback
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
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    total_start = time.perf_counter()
    retrieval_duration_ms = 0.0
    reranking_duration_ms = 0.0
    generation_duration_ms = 0.0
    retrieved_documents = 0
    reranked_documents = 0
    query_status = "success"
    error_type = None

    response_data = None
    response_error = None

    try:
        question = request.question

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

        if not candidates:
            query_status = "no_documents"

            response_data = {
                "answer": (
                    "I could not find supporting documents "
                    "to answer this question."
                ),
                "sources": [],
                "citations_valid": True,
                "has_citations": False,
                "missing_citations": False,
                "answer_quality": {
                    "is_acceptable": False,
                    "has_answer": True,
                    "has_citations": False,
                    "has_sources": False,
                    "is_abstention": True,
                    "issues": ["no_sources"],
                },
            }
        else:
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

            response_data = {
                "answer": result["answer"],
                "sources": result["sources"],
                "citations_valid": validation["is_valid"],
                "has_citations": validation["has_citations"],
                "missing_citations": validation["missing_citations"],
                "answer_quality": quality,
            }

    except Exception as exc:
        query_status = "error"
        error_type = type(exc).__name__
        response_error = exc

        logger.exception("Knowledge-base query failed")

    finally:
        total_duration_ms = (
            time.perf_counter() - total_start
        ) * 1000

        try:
            metric = record_query_metric(
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
            logger.exception("Query performance monitoring failed")
            metric = {"request_id": "unavailable"}

    if response_error is not None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The query could not be processed. "
                "Check the application logs for details."
            ),
        ) from response_error

    return QueryResponse(
        query_id=metric["request_id"],
        **response_data,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def create_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        result = submit_feedback(
            query_id=request.query_id,
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            comment=request.comment,
        )

        logger.info(
            "Feedback submitted: query_id=%s, rating=%s",
            request.query_id,
            request.rating,
        )

        return FeedbackResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Feedback submission failed")

        raise HTTPException(
            status_code=500,
            detail="Feedback could not be saved.",
        ) from exc
