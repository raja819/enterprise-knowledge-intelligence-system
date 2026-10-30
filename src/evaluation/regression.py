from __future__ import annotations
from typing import Any
import math

DEFAULT_TOLERANCE = 0.05

REQUIRED_TOP_LEVEL_METRICS = (
    "pass_rate",
    "answerable_pass_rate",
    "abstention_success_rate",
)

REQUIRED_RETRIEVAL_METRICS = (
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "reciprocal_rank",
)


def _validate_rate(value: Any, metric_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{metric_name} must be a number.")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{metric_name} must be finite.")

    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{metric_name} must be between 0 and 1.")

    return result


def compare_evaluations(
    baseline: dict,
    current: dict,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """
    Compare current RAG evaluation metrics against a saved baseline.

    A regression is reported when a metric drops by more than
    the configured absolute tolerance.
    """

    if not isinstance(baseline, dict) or not isinstance(current, dict):
        raise ValueError("Baseline and current reports must be dictionaries.")

    tolerance = _validate_rate(tolerance, "tolerance")

    baseline_retrieval = baseline.get("retrieval_metrics")
    current_retrieval = current.get("retrieval_metrics")

    if not isinstance(baseline_retrieval, dict):
        raise ValueError("Baseline retrieval_metrics must be an object.")

    if not isinstance(current_retrieval, dict):
        raise ValueError("Current retrieval_metrics must be an object.")

    metric_names = (
        list(REQUIRED_TOP_LEVEL_METRICS)
        + list(REQUIRED_RETRIEVAL_METRICS)
    )

    comparisons = {}
    regressions = []

    for metric_name in metric_names:
        if metric_name in REQUIRED_TOP_LEVEL_METRICS:
            baseline_value = _validate_rate(
                baseline.get(metric_name), f"baseline.{metric_name}"
            )
            current_value = _validate_rate(
                current.get(metric_name), f"current.{metric_name}"
            )
        else:
            baseline_value = _validate_rate(
                baseline_retrieval.get(metric_name),
                f"baseline.retrieval_metrics.{metric_name}",
            )
            current_value = _validate_rate(
                current_retrieval.get(metric_name),
                f"current.retrieval_metrics.{metric_name}",
            )

        change = current_value - baseline_value
        is_regression = change < -tolerance

        comparisons[metric_name] = {
            "baseline": baseline_value,
            "current": current_value,
            "change": change,
            "regression": is_regression,
        }

        if is_regression:
            regressions.append(metric_name)

    return {
        "passed": len(regressions) == 0,
        "tolerance": tolerance,
        "regression_count": len(regressions),
        "regressions": regressions,
        "comparisons": comparisons,
    }
