import importlib.util
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = PROJECT_ROOT / "app" / "observability_dashboard.py"


def load_dashboard_module():
    spec = importlib.util.spec_from_file_location(
        "observability_dashboard", DASHBOARD_PATH
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_file_exists():
    assert DASHBOARD_PATH.is_file()


def test_evaluation_json_loader_reads_valid_json(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "evaluation.json"
    payload = {"total_cases": 8, "passed": 8, "failed": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.load_evaluation_json(str(path), path.stat().st_mtime)

    assert result["total_cases"] == 8
    assert result["passed"] == 8
    assert result["failed"] == 0


def test_evaluation_json_loader_returns_empty_dict_for_missing_file(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "missing.json"

    result = module.load_evaluation_json(str(path), 0)

    assert result == {}


def test_evaluation_csv_loader_reads_rows(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"question": "Question A", "pass": True},
            {"question": "Question B", "pass": False},
        ]
    ).to_csv(path, index=False)

    result = module.load_evaluation_csv(str(path), path.stat().st_mtime)

    assert len(result) == 2
    assert "question" in result.columns
    assert "pass" in result.columns


def test_evaluation_csv_loader_returns_empty_dataframe_for_missing_file(
    tmp_path,
):
    module = load_dashboard_module()
    path = tmp_path / "missing.csv"

    result = module.load_evaluation_csv(str(path), 0)

    assert result.empty


def test_ingestion_loader_reads_jsonl_records(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "ingestion.jsonl"
    records = [
        {"file": "report.pdf", "status": "success", "chunks_indexed": 6},
        {"file": "bad.pdf", "status": "failed", "chunks_indexed": 0},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = module.load_ingestion_logs(str(path), path.stat().st_mtime)

    assert len(result) == 2
    assert result.iloc[0]["status"] == "success"
    assert result.iloc[0]["chunks_indexed"] == 6


def test_ingestion_loader_handles_invalid_jsonl_line(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "ingestion.jsonl"
    path.write_text(
        '{"file": "report.pdf", "status": "success"}\n'
        '{invalid json}\n',
        encoding="utf-8",
    )

    result = module.load_ingestion_logs(str(path), path.stat().st_mtime)

    assert len(result) == 2
    assert result.iloc[1]["status"] == "invalid_log_record"
    assert result.iloc[1]["file"] == "Line 2"


def test_ingestion_loader_returns_empty_dataframe_for_missing_file(tmp_path):
    module = load_dashboard_module()
    path = tmp_path / "missing.jsonl"

    result = module.load_ingestion_logs(str(path), 0)

    assert result.empty
