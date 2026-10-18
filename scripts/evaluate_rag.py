import json
import logging
import re
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.generation.answer_generator import generate_answer
from src.generation.citation_validator import validate_citations
from src.logging_config import setup_logging
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results

logger = logging.getLogger(__name__)

TEST_CASES_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_test_cases.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_results.json"
)

ABSTENTION_MARKERS = (
    "could not find",
    "couldn't find",
    "insufficient evidence",
    "not enough information",
    "does not provide",
    "do not contain enough information",
    "does not contain enough information",
    "cannot determine",
    "can't determine",
    "i cannot determine",
    "i don't have enough information",
    "unable to determine",
    "not mentioned in the provided document excerpts",
    "no answer can be given",
)


def normalize(text: str) -> str:
    """Normalize case and whitespace for answer comparisons."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def answer_matches(expected: str, answer: str) -> bool:
    """
    Check whether the expected answer occurs as a complete phrase.

    This is a lightweight matching heuristic, not a semantic
    or factual correctness evaluator.
    """
    expected_normalized = normalize(expected)
    answer_normalized = normalize(answer)

    if not expected_normalized:
        return False

    pattern = re.escape(expected_normalized)

    # Avoid matching a phrase inside a larger word.
    pattern = rf"(?<!\w){pattern}(?!\w)"

    return re.search(pattern, answer_normalized) is not None


def is_abstention_answer(answer: str) -> bool:
    """Check whether the answer explicitly indicates insufficient evidence."""
    normalized_answer = normalize(answer)

    return any(
        marker in normalized_answer
        for marker in ABSTENTION_MARKERS
    )


def evaluate_case(case: dict) -> dict:
    question = case["question"]
    answerable = case["answerable"]

    candidates = hybrid_search(
        question,
        limit=settings.retrieval_top_k,
    )

    ranked = rerank_results(
        question,
        candidates,
        top_k=settings.rerank_top_k,
    )

    result = generate_answer(question, ranked)
    answer = result["answer"]
    sources = result["sources"]

    output = {
        "question": question,
        "answerable": answerable,
        "answer": answer,
        "retrieved_chunks": len(candidates),
        "ranked_chunks": len(ranked),
    }

    if not answerable:
        abstained = is_abstention_answer(answer)

        output.update({
            "abstained": abstained,
            "pass": abstained,
            "failure_reasons": (
                [] if abstained else ["no_explicit_abstention"]
            ),
        })

        return output

    expected_answer = case["expected_answer"]
    answer_match = answer_matches(expected_answer, answer)

    expected_source = case["expected_source"]
    expected_page = case["expected_page"]

    source_match = any(
        source.get("source") == expected_source
        and source.get("page") == expected_page
        for source in sources
    )

    citation_check = validate_citations(answer, sources)

    failure_reasons = []

    if not answer_match:
        failure_reasons.append("expected_answer_not_found")

    if not source_match:
        failure_reasons.append("expected_source_page_not_found")

    if not citation_check["is_valid"]:
        failure_reasons.append("invalid_citations")

    if not citation_check["has_citations"]:
        failure_reasons.append("missing_citations")

    passed = not failure_reasons

    output.update({
        "expected_answer": expected_answer,
        "expected_source": expected_source,
        "expected_page": expected_page,
        "answer_match": answer_match,
        "expected_source_found": source_match,
        "citations_valid": citation_check["is_valid"],
        "has_citations": citation_check["has_citations"],
        "pass": passed,
        "failure_reasons": failure_reasons,
    })

    return output


def calculate_rate(values: list[bool]) -> float | None:
    """Calculate a rate, returning None when no cases apply."""
    if not values:
        return None

    return round(sum(values) / len(values), 4)


def build_summary(results: list[dict]) -> dict:
    """Build overall and category-specific evaluation metrics."""
    passed = sum(bool(item.get("pass")) for item in results)
    total = len(results)

    answerable_results = [
        item for item in results
        if item.get("answerable") is True
    ]

    unanswerable_results = [
        item for item in results
        if item.get("answerable") is False
    ]

    return {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": calculate_rate([
            bool(item.get("pass")) for item in results
        ]),
        "answerable_cases": len(answerable_results),
        "answerable_pass_rate": calculate_rate([
            bool(item.get("pass")) for item in answerable_results
        ]),
        "answer_match_rate": calculate_rate([
            bool(item.get("answer_match"))
            for item in answerable_results
            if "answer_match" in item
        ]),
        "expected_source_page_match_rate": calculate_rate([
            bool(item.get("expected_source_found"))
            for item in answerable_results
            if "expected_source_found" in item
        ]),
        "citation_validity_rate": calculate_rate([
            bool(item.get("citations_valid"))
            for item in answerable_results
            if "citations_valid" in item
        ]),
        "citation_presence_rate": calculate_rate([
            bool(item.get("has_citations"))
            for item in answerable_results
            if "has_citations" in item
        ]),
        "unanswerable_cases": len(unanswerable_results),
        "abstention_success_rate": calculate_rate([
            bool(item.get("abstained"))
            for item in unanswerable_results
            if "abstained" in item
        ]),
        "results": results,
    }


def main() -> None:
    setup_logging()

    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError(
            "Evaluation dataset must be a non-empty JSON list."
        )

    results = []

    for index, case in enumerate(test_cases, start=1):
        logger.info(
            "Evaluating test case %s/%s",
            index,
            len(test_cases),
        )

        try:
            result = evaluate_case(case)
        except Exception as exc:
            logger.exception("Evaluation failed for case %s", index)

            result = {
                "question": case.get("question", ""),
                "answerable": case.get("answerable"),
                "pass": False,
                "failure_reasons": ["evaluation_error"],
                "error": str(exc),
            }

        results.append(result)

        print(f"\nTest case {index}: {result['question']}")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    summary = build_summary(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nEVALUATION SUMMARY")
    print(f"Passed: {summary['passed']}/{summary['total_cases']}")

    if summary["pass_rate"] is not None:
        print(f"Pass rate: {summary['pass_rate']:.0%}")

    print(
        "Answerable pass rate:",
        summary["answerable_pass_rate"],
    )
    print(
        "Answer match rate:",
        summary["answer_match_rate"],
    )
    print(
        "Source/page match rate:",
        summary["expected_source_page_match_rate"],
    )
    print(
        "Citation validity rate:",
        summary["citation_validity_rate"],
    )
    print(
        "Abstention success rate:",
        summary["abstention_success_rate"],
    )
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
