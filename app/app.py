
import streamlit as st
import sys
import os

# --------------------------------------------------
# PATH
# --------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from rag_backend import build_retrievers, hybrid_search


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# LOAD RETRIEVAL SYSTEM
# --------------------------------------------------

@st.cache_resource
def load_retrieval_system():

    chunks, vector_store, bm25 = build_retrievers()

    return chunks, vector_store, bm25


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧠 Agentic RAG Research Assistant")

st.markdown(
    """
    **Knowledge-grounded research assistant powered by
    Agentic Retrieval-Augmented Generation**
    """
)

st.divider()


# --------------------------------------------------
# LOAD SYSTEM
# --------------------------------------------------

with st.spinner("Loading research document and retrieval system..."):

    chunks, vector_store, bm25 = load_retrieval_system()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Research Workspace")

    st.success("Document Loaded")

    st.write("📄 **r1.pdf**")
    st.write("📑 **19 pages**")

    st.divider()

    st.header("🔎 Retrieval")

    st.write("✅ FAISS")
    st.write("✅ BM25")
    st.write("✅ Hybrid Retrieval")

    st.divider()

    st.header("🤖 Agents")

    st.write("🧠 Planner")
    st.write("✍️ Researcher")
    st.write("🧐 Critic")
    st.write("📚 Citation Verifier")


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

st.subheader("🔍 Research Query")

question = st.text_area(
    "What would you like to research?",
    placeholder=(
        "Example: How does Retrieval-Augmented Generation "
        "improve knowledge-intensive NLP tasks?"
    ),
    height=120
)


research_button = st.button(
    "🚀 Start Research",
    use_container_width=True
)


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

if research_button:

    if not question.strip():

        st.warning("Please enter a research question.")

    else:

        st.success("Research request received.")

        st.divider()

        # ------------------------------------------
        # RETRIEVE
        # ------------------------------------------

        with st.spinner("🔎 Searching research evidence..."):

            retrieved_docs = hybrid_search(
                question,
                vector_store,
                bm25,
                chunks,
                k=3
            )

        st.subheader("🔎 Retrieval Complete")

        st.write(
            f"Retrieved **{len(retrieved_docs)}** relevant passages."
        )

        # ------------------------------------------
        # AGENT PIPELINE
        # ------------------------------------------

        st.subheader("⚙️ Agent Pipeline")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.success("🧠 Planner")

        with col2:
            st.success("🔎 Retriever")

        with col3:
            st.info("✍️ Researcher")

        with col4:
            st.info("🧐 Critic")

        with col5:
            st.info("📚 Citation")

        st.divider()

        # ------------------------------------------
        # ANSWER PLACEHOLDER
        # ------------------------------------------

        st.subheader("🧠 Research Answer")

        st.info(
            "The Gemini-powered research answer will be connected "
            "here after the API quota resets."
        )

        st.divider()

        # ------------------------------------------
        # EVIDENCE
        # ------------------------------------------

        st.subheader("📚 Retrieved Evidence")

        for i, doc in enumerate(retrieved_docs, 1):

            with st.container(border=True):

                st.markdown(
                    f"### Evidence {i}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"📄 **Source:** "
                        f"{doc.metadata.get('source')}"
                    )

                with col2:
                    st.write(
                        f"📑 **Page:** "
                        f"{doc.metadata.get('page')}"
                    )

                with col3:
                    st.write(
                        f"🔢 **Chunk:** "
                        f"{doc.metadata.get('chunk_id')}"
                    )

                st.write(
                    doc.page_content
                )

        st.divider()

        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        st.subheader("🔗 Sources")

        for doc in retrieved_docs:

            st.write(
                f"📄 {doc.metadata.get('source')} "
                f"— Page {doc.metadata.get('page')}"
            )
