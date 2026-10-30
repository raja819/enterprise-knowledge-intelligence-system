import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "baseline" / "rag_baseline.json"
)

CURRENT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_evaluation_results.json"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.regression import compare_evaluations


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(f"Invalid evaluation report: {path}")

    return report


def main() -> int:
    try:
        baseline = load_report(BASELINE_PATH)
        current = load_report(CURRENT_PATH)

        result = compare_evaluations(baseline, current)

        print("\nRAG REGRESSION REPORT")
        print("=" * 60)
        print(f"Baseline: {BASELINE_PATH.name}")
        print(f"Current:  {CURRENT_PATH.name}")
        print(f"Allowed metric drop: {result['tolerance']:.0%}")
        print("-" * 60)

        print(
            f"{'Metric':<28}"
            f"{'Baseline':>10}"
            f"{'Current':>10}"
            f"{'Change':>10}"
        )

        print("-" * 60)

        for name, values in result["comparisons"].items():
            print(
                f"{name:<28}"
                f"{values['baseline']:>10.3f}"
                f"{values['current']:>10.3f}"
                f"{values['change']:>+10.3f}"
            )

        print("-" * 60)

        if result["passed"]:
            print("RESULT: PASSED - No significant regression detected.")
            return 0

        print("RESULT: FAILED - Regression detected.")

        print("Affected metrics:")

        for metric in result["regressions"]:
            print(f"  - {metric}")

        return 1

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
