import json

from src.config import PROJECT_ROOT


TEST_CASES_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_test_cases.json"
)


def test_evaluation_dataset_exists():
    assert TEST_CASES_PATH.exists()


def test_evaluation_dataset_is_valid():
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    assert isinstance(cases, list)
    assert len(cases) >= 4

    for case in cases:
        assert isinstance(case["question"], str)
        assert case["question"].strip()
        assert isinstance(case["answerable"], bool)


def test_answerable_cases_have_expected_answers():
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    answerable_cases = [
        case for case in cases if case["answerable"]
    ]

    assert answerable_cases

    for case in answerable_cases:
        assert case["expected_answer"].strip()
        assert case["expected_source"].strip()
        assert isinstance(case["expected_page"], int)


def test_unanswerable_cases_are_present():
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    unanswerable_cases = [
        case for case in cases if not case["answerable"]
    ]

    assert unanswerable_cases
