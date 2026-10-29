
import pytest

from src.evaluation.answer_quality import evaluate_answer_quality


SOURCES = [
    {
        "citation": "S1",
        "source": "sample_report.pdf",
        "page": 1,
    }
]


def test_accepts_answer_with_known_citation():
    result = evaluate_answer_quality(
        "Revenue was ₹12.5 crore. [S1]",
        SOURCES,
    )

    assert result["is_acceptable"] is True
    assert result["has_citations"] is True
    assert result["unknown_citations"] == []


def test_rejects_answer_with_unknown_citation():
    result = evaluate_answer_quality(
        "Revenue was ₹12.5 crore. [S9]",
        SOURCES,
    )

    assert result["is_acceptable"] is False
    assert result["unknown_citations"] == ["S9"]
    assert "unknown_citations" in result["issues"]


def test_rejects_answer_without_citation():
    result = evaluate_answer_quality(
        "Revenue was ₹12.5 crore.",
        SOURCES,
    )

    assert result["is_acceptable"] is False
    assert "missing_citations" in result["issues"]


def test_rejects_answer_without_sources():
    result = evaluate_answer_quality(
        "Revenue was ₹12.5 crore. [S1]",
        [],
    )

    assert result["is_acceptable"] is False
    assert "no_sources" in result["issues"]


def test_detects_abstention():
    result = evaluate_answer_quality(
        "The supplied documents do not contain enough information to answer. [S1]",
        SOURCES,
    )

    assert result["is_abstention"] is True
    assert result["is_acceptable"] is True


def test_rejects_empty_answer():
    result = evaluate_answer_quality("", SOURCES)

    assert result["is_acceptable"] is False
    assert "empty_answer" in result["issues"]


def test_rejects_non_string_answer():
    with pytest.raises(ValueError):
        evaluate_answer_quality(None, SOURCES)


def test_accepts_abstention_without_citation_when_sources_exist():
    result = evaluate_answer_quality(
        "The supplied documents do not contain enough information "
        "to determine the revenue in 2030.",
        SOURCES,
    )

    assert result["is_abstention"] is True
    assert result["has_citations"] is False
    assert result["is_acceptable"] is True


def test_mixed_answer_with_citation_is_not_classified_as_plain_abstention():
    answer = (
        "The report does not contain enough information. "
        "Revenue was ₹50 crore. [S1]"
    )

    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = evaluate_answer_quality(answer, sources)

    assert result["is_abstention"] is False
    assert result["has_citations"] is True


def test_unknown_citation_does_not_make_answer_acceptable():
    answer = "Revenue was ₹50 crore. [S9]"

    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = evaluate_answer_quality(answer, sources)

    assert result["is_acceptable"] is False
    assert result["unknown_citations"] == ["S9"]
