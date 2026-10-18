
import logging

from fastapi import FastAPI, HTTPException

from api.schemas import QueryRequest, QueryResponse
from src.config import settings
from src.generation.answer_generator import generate_answer
from src.generation.citation_validator import validate_citations
from src.logging_config import setup_logging
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.evaluation.answer_quality import evaluate_answer_quality

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
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        candidates = hybrid_search(
            question,
            limit=settings.retrieval_top_k,
        )

        # Case 1: No relevant documents were retrieved.
        if not candidates:
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
        ranked_documents = rerank_results(
            question,
            candidates,
            top_k=settings.rerank_top_k,
        )

        # Case 3: Generate an answer using the local LLM.
        result = generate_answer(
            question,
            ranked_documents,
        )

        # Case 4: Validate citation labels.
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
            len(candidates),
            len(ranked_documents),
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

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Knowledge-base query failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "The query could not be processed. "
                "Check the application logs for details."
            ),
        ) from exc
