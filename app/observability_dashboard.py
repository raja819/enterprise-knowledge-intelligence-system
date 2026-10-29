import json
from pathlib import Path

import pandas as pd
import streamlit as st


# Project paths: this file is inside the app/ directory.
ROOT = Path(__file__).resolve().parent.parent
EVALUATION_JSON = ROOT / "data" / "evaluation" / "rag_evaluation_results.json"
EVALUATION_CSV = ROOT / "data" / "evaluation" / "rag_evaluation_results.csv"
INGESTION_LOG = ROOT / "logs" / "ingestion_audit.jsonl"

st.set_page_config(
    page_title="RAG Observability Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("RAG Observability & Evaluation Dashboard")
st.caption(
    "Monitor evaluation quality, retrieval performance, "
    "question-level results, and document ingestion."
)


@st.cache_data
def load_evaluation_json(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_evaluation_csv(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_ingestion_logs(path_string, modified_time):
    path = Path(path_string)
    records = []

    if not path.exists():
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except json.JSONDecodeError:
                records.append(
                    {
                        "status": "invalid_log_record",
                        "file": f"Line {line_number}",
                    }
                )

    return pd.DataFrame(records)


def file_mtime(path):
    return path.stat().st_mtime if path.exists() else 0


evaluation = load_evaluation_json(
    str(EVALUATION_JSON), file_mtime(EVALUATION_JSON)
)
results = load_evaluation_csv(
    str(EVALUATION_CSV), file_mtime(EVALUATION_CSV)
)
ingestion = load_ingestion_logs(
    str(INGESTION_LOG), file_mtime(INGESTION_LOG)
)

with st.sidebar:
    st.header("Dashboard controls")
    if st.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Data is read from local evaluation and ingestion files.")
    st.caption("The dashboard does not run model training or ingestion.")

st.header("1. Evaluation overview")

if not evaluation:
    st.warning(
        "Evaluation JSON not found. Run the project's RAG evaluation "
        "script before reviewing evaluation metrics."
    )
else:
    total = int(evaluation.get("total_cases", 0))
    passed = int(evaluation.get("passed", 0))
    failed = int(evaluation.get("failed", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total test cases", total)
    c2.metric("Passed", passed)
    c3.metric("Failed", failed)
    c4.metric(
        "Pass rate",
        f"{float(evaluation.get('pass_rate', 0)):.1%}",
    )

    c5, c6, c7 = st.columns(3)
    c5.metric(
        "Answerable cases",
        int(evaluation.get("answerable_cases", 0)),
    )
    c6.metric(
        "Answerable pass rate",
        f"{float(evaluation.get('answerable_pass_rate', 0)):.1%}",
    )
    c7.metric(
        "Abstention success rate",
        f"{float(evaluation.get('abstention_success_rate', 0)):.1%}",
    )

    generated_at = evaluation.get("generated_at_utc")
    if generated_at:
        st.caption(f"Evaluation generated at (UTC): {generated_at}")

st.divider()
st.header("2. Retrieval performance")

metrics = evaluation.get("retrieval_metrics", {}) if evaluation else {}

if metrics:
    metric_names = {
        "precision_at_1": "Precision@1",
        "precision_at_3": "Precision@3",
        "precision_at_5": "Precision@5",
        "recall_at_1": "Recall@1",
        "recall_at_3": "Recall@3",
        "recall_at_5": "Recall@5",
        "reciprocal_rank": "Mean Reciprocal Rank",
    }

    metric_rows = [
        {"Metric": label, "Score": float(metrics[key])}
        for key, label in metric_names.items()
        if key in metrics and metrics[key] is not None
    ]

    if metric_rows:
        metric_df = pd.DataFrame(metric_rows)
        left, right = st.columns([3, 2])

        with left:
            st.dataframe(
                metric_df.style.format({"Score": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.bar_chart(
                metric_df.set_index("Metric"),
                y="Score",
            )
else:
    st.info("No retrieval metrics are available in the evaluation JSON.")

definition = (
    evaluation.get("retrieval_metrics_definition")
    if evaluation else None
)
if definition:
    with st.expander("Retrieval metric definitions"):
        st.json(definition)

st.divider()
st.header("3. Test-case explorer")

if results.empty:
    st.warning(
        "Evaluation CSV not found or empty. Generate evaluation results "
        "to explore individual questions."
    )
else:
    filtered = results.copy()

    if "pass" in filtered.columns:
        outcome = st.selectbox(
            "Filter by outcome",
            ["All", "Passed", "Failed"],
        )
        if outcome == "Passed":
            filtered = filtered[filtered["pass"].astype(bool)]
        elif outcome == "Failed":
            filtered = filtered[~filtered["pass"].astype(bool)]

    if "answerable" in filtered.columns:
        answerability = st.selectbox(
            "Filter by expected answerability",
            ["All", "Answerable", "Unanswerable"],
        )
        if answerability == "Answerable":
            filtered = filtered[filtered["answerable"].astype(bool)]
        elif answerability == "Unanswerable":
            filtered = filtered[~filtered["answerable"].astype(bool)]

    if "question" in filtered.columns:
        search_text = st.text_input("Search questions")
        if search_text.strip():
            filtered = filtered[
                filtered["question"].astype(str).str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

    st.caption(f"Showing {len(filtered)} of {len(results)} test cases.")

    display_columns = [
        column for column in [
            "question",
            "answerable",
            "pass",
            "failure_reasons",
            "answer_match",
            "expected_source_found",
            "citations_valid",
            "abstained",
            "reciprocal_rank",
        ]
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    if "pass" in filtered.columns and not filtered.empty:
        failed_rows = filtered[~filtered["pass"].astype(bool)]
        if not failed_rows.empty:
            st.subheader("Failure analysis")
            for _, row in failed_rows.iterrows():
                question = row.get("question", "Unknown question")
                reasons = row.get("failure_reasons", "No reason recorded")
                with st.expander(str(question)):
                    st.write("Failure reasons:", reasons)
                    st.json(row.to_dict())

st.divider()
st.header("4. Document ingestion monitoring")

if ingestion.empty:
    st.info(
        "No ingestion audit records found. Run document ingestion "
        "to populate the local audit log."
    )
else:
    total_runs = len(ingestion)
    if "status" in ingestion.columns:
        successful = int((ingestion["status"] == "success").sum())
        unsuccessful = total_runs - successful
    else:
        successful = 0
        unsuccessful = total_runs

    c1, c2, c3 = st.columns(3)
    c1.metric("Audit records", total_runs)
    c2.metric("Successful records", successful)
    c3.metric("Other statuses", unsuccessful)

    if "duration_seconds" in ingestion.columns:
        durations = pd.to_numeric(
            ingestion["duration_seconds"], errors="coerce"
        )
        st.metric(
            "Average ingestion duration",
            f"{durations.mean():.2f} seconds"
            if durations.notna().any()
            else "N/A",
        )

    st.dataframe(
        ingestion,
        use_container_width=True,
        hide_index=True,
    )

    numeric_columns = [
        column for column in [
            "pages_extracted",
            "chunks_created",
            "chunks_indexed",
            "duration_seconds",
        ]
        if column in ingestion.columns
    ]

    if numeric_columns:
        chart_data = ingestion[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        st.subheader("Ingestion volume and duration")
        st.line_chart(chart_data)

st.divider()
st.caption(
    "Interpretation note: evaluation scores reflect the configured test "
    "cases and relevance labels; they do not by themselves guarantee "
    "production accuracy."
)
