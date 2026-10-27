import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app


TEST_CASES_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_test_cases.json"
)


def normalize_answer(text: str) -> str:
    """Normalize text for robust expected-answer comparisons."""
    text = text.lower()
    text = re.sub(r"[^\w.\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def main() -> int:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    answerable_cases = [case for case in cases if case["answerable"]]
    unanswerable_cases = [
        case for case in cases if not case["answerable"]
    ]

    answerable_passes = 0
    abstention_passes = 0

    print(f"Evaluation cases: {len(cases)}")
    print(f"Answerable: {len(answerable_cases)}")
    print(f"Unanswerable: {len(unanswerable_cases)}")
    print("-" * 60)

    with TestClient(app) as client:
        for case in cases:
            question = case["question"]
            response = client.post(
                "/query",
                json={"question": question},
            )

            if response.status_code != 200:
                print(f"[FAIL] HTTP {response.status_code}: {question}")
                continue

            body = response.json()
            answer = body["answer"]
            sources = body["sources"]

            if case["answerable"]:
                expected = normalize_answer(case["expected_answer"])
                actual = normalize_answer(answer)

                answer_found = expected in actual
                source_found = any(
                    source.get("source") == case["expected_source"]
                    and source.get("page") == case["expected_page"]
                    for source in sources
                )
                citations_valid = body["citations_valid"]

                passed = (
                    answer_found
                    and source_found
                    and citations_valid
                )

                if passed:
                    answerable_passes += 1

                print(
                    f"[{'PASS' if passed else 'FAIL'}] "
                    f"Answerable: {question}"
                )
                print(
                    f"       expected_answer_found={answer_found}, "
                    f"expected_source_page_found={source_found}, "
                    f"citations_valid={citations_valid}"
                )

            else:
                quality = body.get("answer_quality", {})
                abstained = quality.get("is_abstention") is True

                if abstained:
                    abstention_passes += 1

                print(
                    f"[{'PASS' if abstained else 'FAIL'}] "
                    f"Unanswerable: {question}"
                )
                print(
                    f"       abstained={abstained}, "
                    f"sources_returned={len(sources)}"
                )

                # Diagnostic details for failed abstention checks.
                if not abstained:
                    print(f"       answer={answer}")
                    print(
                        "       citations_valid="
                        f"{body.get('citations_valid')}"
                    )
                    print(
                        "       missing_citations="
                        f"{body.get('missing_citations')}"
                    )
                    print(
                        "       answer_quality={quality}".format(
                            quality=quality
                        )
                    )

    answerable_total = len(answerable_cases)
    unanswerable_total = len(unanswerable_cases)

    answerable_rate = (
        answerable_passes / answerable_total
        if answerable_total else 0.0
    )
    refusal_rate = (
        abstention_passes / unanswerable_total
        if unanswerable_total else 0.0
    )

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print(f"Answerable checks: {answerable_passes}/{answerable_total}")
    print(f"Answerable pass rate: {answerable_rate:.1%}")
    print(
        f"Unanswerable refusals: "
        f"{abstention_passes}/{unanswerable_total}"
    )
    print(f"Unanswerable refusal rate: {refusal_rate:.1%}")

    all_passed = (
        answerable_passes == answerable_total
        and abstention_passes == unanswerable_total
    )

    print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
