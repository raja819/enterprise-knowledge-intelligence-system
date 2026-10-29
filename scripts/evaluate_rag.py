import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app
from src.retrieval.hybrid_search import (
    get_indexed_chunks,
    hybrid_search,
)
from src.retrieval.reranker import rerank_results

TEST_CASES_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_test_cases.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
JSON_REPORT_PATH = OUTPUT_DIR / "rag_evaluation_results.json"
CSV_REPORT_PATH = OUTPUT_DIR / "rag_evaluation_results.csv"

K_VALUES = (1, 3, 5)


def normalize_answer(text: str) -> str:
    """Normalize text for expected-answer comparisons."""
    text = text.lower()
    text = re.sub(r"[^\w.\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def is_relevant(document: dict, case: dict) -> bool:
    """Match a retrieved chunk to the expected source and page."""
    metadata = document.get("metadata", {})
    return (
        metadata.get("source") == case["expected_source"]
        and metadata.get("page") == case["expected_page"]
    )


def calculate_retrieval_metrics(
    ranked_documents: list[dict],
    relevant_chunk_count: int,
    case: dict,
) -> dict:
    """Calculate chunk-level Precision@K, Recall@K and reciprocal rank."""
    metrics = {}

    relevant_ranks = [
        rank
        for rank, document in enumerate(ranked_documents, start=1)
        if is_relevant(document, case)
    ]

    first_relevant_rank = (
        relevant_ranks[0] if relevant_ranks else None
    )

    metrics["first_relevant_rank"] = first_relevant_rank
    metrics["reciprocal_rank"] = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    for k in K_VALUES:
        top_k = ranked_documents[:k]
        relevant_retrieved = sum(
            is_relevant(document, case)
            for document in top_k
        )

        metrics[f"precision_at_{k}"] = (
            relevant_retrieved / len(top_k)
            if top_k else 0.0
        )

        metrics[f"recall_at_{k}"] = (
            relevant_retrieved / relevant_chunk_count
            if relevant_chunk_count > 0 else 0.0
        )

    return metrics


def main() -> int:
    with TEST_CASES_PATH.open("r", encoding="utf-8-sig") as file:
        cases = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indexed_chunks = get_indexed_chunks()
    evaluation_results = []

    answerable_cases = [case for case in cases if case["answerable"]]
    unanswerable_cases = [case for case in cases if not case["answerable"]]

    print(f"Evaluation cases: {len(cases)}")
    print(f"Answerable: {len(answerable_cases)}")
    print(f"Unanswerable: {len(unanswerable_cases)}")
    print("-" * 60)

    with TestClient(app) as client:
        for case in cases:
            question = case["question"]
            answerable = case["answerable"]

            result = {
                "question": question,
                "answerable": answerable,
                "pass": False,
                "failure_reasons": [],
            }

            # Evaluate retrieval separately from the answer API.
            # These direct calls expose ranked chunks and their metadata.
            ranked_documents = []
            retrieval_metrics = {}

            if answerable:
                candidates = hybrid_search(question)
                ranked_documents = rerank_results(
                    question,
                    candidates,
                )

                relevant_chunk_count = sum(
                    1
                    for chunk in indexed_chunks
                    if is_relevant(chunk, case)
                )

                retrieval_metrics = calculate_retrieval_metrics(
                    ranked_documents,
                    relevant_chunk_count,
                    case,
                )

                result.update(retrieval_metrics)
                result["relevant_chunks_in_index"] = relevant_chunk_count
                result["retrieved_chunks"] = len(candidates)
                result["ranked_chunks"] = len(ranked_documents)

                if relevant_chunk_count == 0:
                    result["failure_reasons"].append(
                        "expected_source_page_not_in_index"
                    )

            response = client.post(
                "/query",
                json={"question": question},
            )

            if response.status_code != 200:
                result["http_status"] = response.status_code
                result["failure_reasons"].append(
                    f"http_{response.status_code}"
                )
                evaluation_results.append(result)
                print(f"[FAIL] HTTP {response.status_code}: {question}")
                continue

            body = response.json()
            answer = body["answer"]
            sources = body["sources"]

            result.update({
                "http_status": response.status_code,
                "answer": answer,
                "source_count": len(sources),
                "citations_valid": body["citations_valid"],
                "has_citations": body["has_citations"],
            })

            if answerable:
                expected = normalize_answer(case["expected_answer"])
                actual = normalize_answer(answer)

                answer_found = expected in actual
                source_found = any(
                    source.get("source") == case["expected_source"]
                    and source.get("page") == case["expected_page"]
                    for source in sources
                )
                citations_valid = body["citations_valid"]

                result.update({
                    "expected_answer": case["expected_answer"],
                    "expected_source": case["expected_source"],
                    "expected_page": case["expected_page"],
                    "answer_match": answer_found,
                    "expected_source_found": source_found,
                    "retrieval_source_page_match": (
                        retrieval_metrics.get("first_relevant_rank")
                        is not None
                    ),
                })

                if not answer_found:
                    result["failure_reasons"].append(
                        "expected_answer_not_found"
                    )
                if not source_found:
                    result["failure_reasons"].append(
                        "expected_source_page_not_cited"
                    )
                if not citations_valid:
                    result["failure_reasons"].append(
                        "invalid_citations"
                    )
                if relevant_chunk_count == 0:
                    pass
                result["pass"] = (
                    answer_found
                    and source_found
                    and citations_valid
                    and relevant_chunk_count > 0
                )

            else:
                quality = body.get("answer_quality", {})
                abstained = quality.get("is_abstention") is True

                result["abstained"] = abstained

                if not abstained:
                    result["failure_reasons"].append(
                        "expected_abstention_not_detected"
                    )

                result["pass"] = abstained

            evaluation_results.append(result)

            print(
                f"[{'PASS' if result['pass'] else 'FAIL'}] "
                f"{'Answerable' if answerable else 'Unanswerable'}: "
                f"{question}"
            )

            if answerable:
                print(
                    "       "
                    f"answer_match={result.get('answer_match')}, "
                    f"source_match={result.get('expected_source_found')}, "
                    f"MRR={result.get('reciprocal_rank', 0.0):.3f}"
                )
                for k in K_VALUES:
                    print(
                        f"       P@{k}="
                        f"{result.get(f'precision_at_{k}', 0.0):.3f}, "
                        f"R@{k}="
                        f"{result.get(f'recall_at_{k}', 0.0):.3f}"
                    )
            else:
                print(f"       abstained={result.get('abstained')}")

    passed = sum(result["pass"] for result in evaluation_results)
    failed = len(evaluation_results) - passed

    answerable_results = [
        result for result in evaluation_results
        if result["answerable"]
    ]
    unanswerable_results = [
        result for result in evaluation_results
        if not result["answerable"]
    ]

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(evaluation_results),
        "passed": passed,
        "failed": failed,
        "pass_rate": rate(passed, len(evaluation_results)),
        "answerable_cases": len(answerable_results),
        "unanswerable_cases": len(unanswerable_results),
        "answerable_pass_rate": rate(
            sum(r["pass"] for r in answerable_results),
            len(answerable_results),
        ),
        "abstention_success_rate": rate(
            sum(r.get("abstained", False) for r in unanswerable_results),
            len(unanswerable_results),
        ),
        "retrieval_metrics_definition": (
            "Relevant chunk means matching expected_source and "
            "expected_page. Recall denominator is the number of indexed "
            "chunks on that source/page. Metrics are computed for "
            "answerable cases only."
        ),
        "retrieval_metrics": {},
        "results": evaluation_results,
    }

    for metric in [
        *(f"precision_at_{k}" for k in K_VALUES),
        *(f"recall_at_{k}" for k in K_VALUES),
        "reciprocal_rank",
    ]:
        values = [
            result[metric]
            for result in answerable_results
            if metric in result
        ]
        summary["retrieval_metrics"][metric] = (
            sum(values) / len(values) if values else 0.0
        )

    with JSON_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    csv_fields = [
        "question",
        "answerable",
        "pass",
        "failure_reasons",
        "answer_match",
        "expected_source_found",
        "citations_valid",
        "abstained",
        "retrieved_chunks",
        "ranked_chunks",
        "relevant_chunks_in_index",
        "first_relevant_rank",
        "reciprocal_rank",
        *(f"precision_at_{k}" for k in K_VALUES),
        *(f"recall_at_{k}" for k in K_VALUES),
    ]

    with CSV_REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for result in evaluation_results:
            row = result.copy()
            row["failure_reasons"] = "; ".join(
                row.get("failure_reasons", [])
            )
            writer.writerow(row)

    print("\n" + "=" * 60)
    print("ADVANCED RAG EVALUATION SUMMARY")
    print(f"Passed: {passed}/{len(evaluation_results)}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print("\nAverage retrieval metrics (answerable cases):")

    for metric, value in summary["retrieval_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nJSON report: {JSON_REPORT_PATH}")
    print(f"CSV report:  {CSV_REPORT_PATH}")
    print(f"Overall result: {'PASS' if failed == 0 else 'FAIL'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())