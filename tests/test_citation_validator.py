import pytest

from src.generation.citation_validator import validate_citations


def test_valid_citation_is_accepted():
    answer = "Revenue was ₹12.5 crore. [S1]"
    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is True
    assert result["has_citations"] is True
    assert result["unknown_citations"] == []


def test_unknown_citation_is_rejected():
    answer = "Revenue was ₹12.5 crore. [S99]"
    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is False
    assert result["unknown_citations"] == ["S99"]


def test_missing_citation_is_detected():
    answer = "Revenue was ₹12.5 crore."
    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is False
    assert result["has_citations"] is False
    assert result["missing_citations"] is True


def test_invalid_answer_type_is_rejected():
    with pytest.raises(ValueError, match="Answer must be a string"):
        validate_citations(None, [])


def test_missing_citations_make_validation_invalid():
    result = validate_citations(
        "The company earned ₹12.5 crore.",
        [{"citation": "S1", "source": "sample_report.pdf", "page": 1}],
    )

    assert result["is_valid"] is False
    assert result["has_citations"] is False
    assert result["missing_citations"] is True


def test_valid_citation_label_is_detected():
    result = validate_citations(
        "The company earned ₹12.5 crore. [S1]",
        [{"citation": "S1", "source": "sample_report.pdf", "page": 1}],
    )

    assert result["is_valid"] is True
    assert result["cited_labels"] == ["S1"]
    assert result["unknown_citations"] == []
    assert result["missing_citations"] is False


def test_unknown_citation_is_rejected_even_when_other_sources_exist():
    answer = "Revenue was ₹12.5 crore. [S9]"
    sources = [
        {
            "citation": "S1",
            "source": "sample_report.pdf",
            "page": 1,
        }
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is False
    assert result["unknown_citations"] == ["S9"]
    assert result["has_citations"] is True
    assert result["missing_citations"] is False

