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
    "Ask questions about your indexed enterprise documents."
)

st.info(
    "Demo knowledge base: ACME Retail Company's sample report."
)

question = st.text_area(
    "Enter your question",
    placeholder="What was the total revenue in 2025?",
    max_chars=2000,
    height=100,
)

if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching documents and generating answer..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip()},
                    timeout=180,
                )

            if response.ok:
                result = response.json()

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Source references")

                sources = result.get("sources", [])

                if sources:
                    for source in sources:
                        st.markdown(
                            f"- **[{source['citation']}]** "
                            f"`{source['source']}` — "
                            f"Page {source['page']}"
                        )
                else:
                    st.write("No supporting sources were returned.")

                st.caption(
                    f"Citation labels valid: "
                    f"{result.get('citations_valid', False)}"
                )
                st.caption(
                    f"Citations present: "
                    f"{result.get('has_citations', False)}"
                )
            else:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.RequestException as exc:
            st.error(
                "Could not connect to the RAG API. "
                "Make sure the FastAPI server is running at "
                f"{API_URL}. Details: {exc}"
            )
