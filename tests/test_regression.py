import pytest

from src.evaluation.regression import compare_evaluations


@pytest.fixture
def baseline_report():
    return {
        "pass_rate": 1.0,
        "answerable_pass_rate": 1.0,
        "abstention_success_rate": 1.0,
        "retrieval_metrics": {
            "precision_at_1": 0.67,
            "precision_at_3": 0.33,
            "precision_at_5": 0.27,
            "recall_at_1": 0.33,
            "recall_at_3": 0.50,
            "recall_at_5": 0.67,
            "reciprocal_rank": 0.83,
        },
    }


def test_identical_reports_have_no_regression(baseline_report):
    result = compare_evaluations(baseline_report, baseline_report)

    assert result["passed"] is True
    assert result["regression_count"] == 0
    assert result["regressions"] == []


def test_small_metric_drop_within_tolerance_passes(baseline_report):
    current = {
        **baseline_report,
        "retrieval_metrics": {
            **baseline_report["retrieval_metrics"],
            "recall_at_5": 0.64,
        },
    }

    result = compare_evaluations(baseline_report, current)

    assert result["passed"] is True
    assert result["regression_count"] == 0


def test_large_metric_drop_is_detected(baseline_report):
    current = {
        **baseline_report,
        "retrieval_metrics": {
            **baseline_report["retrieval_metrics"],
            "recall_at_5": 0.50,
        },
    }

    result = compare_evaluations(baseline_report, current)

    assert result["passed"] is False
    assert "recall_at_5" in result["regressions"]


def test_multiple_regressions_are_detected(baseline_report):
    current = {
        **baseline_report,
        "pass_rate": 0.80,
        "retrieval_metrics": {
            **baseline_report["retrieval_metrics"],
            "precision_at_1": 0.50,
            "reciprocal_rank": 0.60,
        },
    }

    result = compare_evaluations(baseline_report, current)

    assert result["passed"] is False
    assert result["regression_count"] == 3


def test_custom_tolerance_is_respected(baseline_report):
    current = {
        **baseline_report,
        "retrieval_metrics": {
            **baseline_report["retrieval_metrics"],
            "recall_at_5": 0.60,
        },
    }

    result = compare_evaluations(
        baseline_report,
        current,
        tolerance=0.10,
    )

    assert result["passed"] is True


def test_missing_metric_raises_error(baseline_report):
    current = {
        **baseline_report,
        "retrieval_metrics": {
            **baseline_report["retrieval_metrics"],
        },
    }
    del current["retrieval_metrics"]["recall_at_5"]

    with pytest.raises(ValueError, match="recall_at_5"):
        compare_evaluations(baseline_report, current)


@pytest.mark.parametrize(
    "invalid_value",
    [-0.1, 1.1, "invalid", None, True, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_metric_value_raises_error(
    baseline_report,
    invalid_value,
):
    current = {
        **baseline_report,
        "pass_rate": invalid_value,
    }

    with pytest.raises(ValueError):
        compare_evaluations(baseline_report, current)


def test_invalid_tolerance_raises_error(baseline_report):
    with pytest.raises(ValueError, match="tolerance"):
        compare_evaluations(
            baseline_report,
            baseline_report,
            tolerance=1.5,
        )


def test_non_dictionary_report_raises_error(baseline_report):
    with pytest.raises(ValueError, match="dictionaries"):
        compare_evaluations([], baseline_report)
