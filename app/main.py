import re
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise Knowledge Intelligence System",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise Knowledge Intelligence System")
st.caption(
    "Ask questions about indexed enterprise documents "
    "and inspect the supporting evidence."
)

st.info(
    "Demo knowledge base: ACME Retail Company's sample report."
)

with st.sidebar:
    st.header("System status")
    st.write(f"API endpoint: `{API_URL}`")
    st.caption("FastAPI and Ollama must be running.")

    if st.button("Check API health", use_container_width=True):
        try:
            health_response = requests.get(
                f"{API_URL}/health",
                timeout=10,
            )
            if health_response.ok:
                st.success("API is reachable.")
                st.json(health_response.json())
            else:
                st.error(
                    f"Health check failed: "
                    f"HTTP {health_response.status_code}"
                )
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

st.subheader("Ask a question")

question = st.text_area(
    "Enter your question",
    placeholder="What was ACME Retail's revenue in 2025?",
    max_chars=2000,
    height=100,
)

if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip()},
                    timeout=180,
                )

            if not response.ok:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )
            else:
                result = response.json()
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                quality = result.get("answer_quality", {})

                st.session_state["last_result"] = result
                st.session_state["last_question"] = question.strip()

        except requests.RequestException as exc:
            st.error(
                "Could not connect to the RAG API. "
                "Make sure FastAPI is running. "
                f"Details: {exc}"
            )

result = st.session_state.get("last_result")

if result:
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    quality = result.get("answer_quality", {})

    cited_labels = set(
        re.findall(r"\[(S\d+)\]", answer)
    )
    available_labels = {
        source.get("citation")
        for source in sources
        if source.get("citation")
    }

    is_abstention = quality.get("is_abstention", False)
    citations_valid = result.get("citations_valid", False)
    has_citations = result.get("has_citations", False)

    st.divider()
    st.subheader("Answer")

    if is_abstention:
        st.warning(answer)
    else:
        st.markdown(answer)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Answer quality",
        "Acceptable" if quality.get("is_acceptable") else "Review",
    )
    col2.metric(
        "Citation labels",
        "Valid" if citations_valid else "Not validated",
    )
    col3.metric(
        "Citations in answer",
        "Present" if has_citations else "Absent",
    )

    st.subheader("Source references")

    cited_sources = [
        source
        for source in sources
        if source.get("citation") in cited_labels
    ]

    if cited_sources:
        for source in cited_sources:
            st.markdown(
                f"- **[{source['citation']}]** "
                f"`{source['source']}` — "
                f"Page {source['page']}"
            )
    elif is_abstention:
        st.info(
            "No source is cited in the answer. "
            "The system could not establish the requested fact "
            "from the available evidence."
        )
    else:
        st.warning("No matching cited source was returned.")

    with st.expander("Inspect all retrieved sources"):
        if sources:
            for source in sources:
                label = source.get("citation", "Unknown")
                st.markdown(
                    f"- **[{label}]** "
                    f"`{source.get('source', 'Unknown file')}` — "
                    f"Page {source.get('page', 'Unknown')}"
                )
        else:
            st.write("No sources were returned.")

    with st.expander("Technical validation details"):
        st.json(
            {
                "citations_valid": citations_valid,
                "has_citations": has_citations,
                "missing_citations": result.get(
                    "missing_citations"
                ),
                "answer_quality": quality,
            }
        )
