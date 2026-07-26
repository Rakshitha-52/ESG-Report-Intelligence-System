"""
ESG Report Intelligence System - Streamlit Application

Main entry point. Provides PDF upload, chat-based Q&A, and cited
answers powered by the RAG pipeline (FAISS + Gemini).

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append("src")
from generation.comparison import ComparisonHandler

from ingestion.pdf_loader import PDFLoader
from ingestion.text_preprocessor import TextPreprocessor
from ingestion.chunker import DocumentChunker
from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore
from generation.rag_chain import ESGRAGPipeline
from generation.citation_extractor import CitationExtractor

load_dotenv()

st.set_page_config(
    page_title="ESG Report Intelligence",
    page_icon="🌱",
    layout="wide"
)


# ---------------- Styling ----------------

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="esg-hero">
            <div class="esg-hero-eyebrow">Retrieval-Augmented Analysis · 8 Reports · 5 Companies</div>
            <div class="esg-hero-title">ESG Report Intelligence</div>
            <div class="esg-hero-sub">
                Ask questions across sustainability reports from Apple, Google, Microsoft,
                Reliance, and Tata. Every answer is grounded in the source text and cited
                by document and page.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_citation_chips(citations):
    if not citations:
        st.markdown(
            '<div class="esg-no-citation">⚠ no sources cited — answer may not be grounded</div>',
            unsafe_allow_html=True
        )
        return

    chips = "".join(
        f'<span class="esg-citation-chip">{c["source"]} '
        f'<span class="esg-page">p.{c["page"]}</span></span>'
        for c in citations
    )
    st.markdown(f'<div class="esg-citation-row">{chips}</div>', unsafe_allow_html=True)


# ---------------- Session State ----------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "k_value" not in st.session_state:
    st.session_state.k_value = 5


# ---------------- Cached Resources ----------------

@st.cache_resource
def load_embedding_generator():
    return EmbeddingGenerator()


# ---------------- Pipeline Functions ----------------

def process_uploaded_pdfs(uploaded_files, embedding_generator):
    """Run the full ingestion pipeline on newly uploaded PDFs, with
    defensive error handling for bad files or extraction failures."""
    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.warning(f"Skipped {uploaded_file.name} — not a PDF file.")
            continue

        try:
            with open(pdf_dir / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            valid_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Could not save {uploaded_file.name}: {e}")

    if not valid_files:
        st.error("No valid PDF files to process.")
        return

    try:
        with st.spinner("Extracting text from PDFs..."):
            loader = PDFLoader(str(pdf_dir))
            documents = loader.load_all_pdfs()
    except Exception as e:
        st.error(f"Failed to extract text from PDFs: {e}")
        return

    if not documents:
        st.error(
            "No extractable text found in the uploaded PDF(s). "
            "The file(s) may be scanned images without a text layer."
        )
        return

    with st.spinner("Cleaning text..."):
        preprocessor = TextPreprocessor()
        cleaned_docs = preprocessor.preprocess_documents(documents)

    with st.spinner("Chunking documents..."):
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(cleaned_docs)

    with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = embedding_generator.embed_batch(texts, show_progress=False)

    with st.spinner("Building vector database..."):
        vector_store = FAISSVectorStore(embedding_dim=embedding_generator.embedding_dim)
        vector_store.add_documents(texts, embeddings, metadatas)
        vector_store.save("data/vector_db")

    st.session_state.vector_store = vector_store
    st.session_state.rag_pipeline = ESGRAGPipeline(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )

    st.success(f"✅ Processed {len(valid_files)} PDF(s) into {len(chunks)} chunks.")


def load_existing_database(embedding_generator):
    """Load a previously built vector database from disk, if present.

    This is what powers both the sidebar's 'Load Existing Database'
    button and the automatic startup load in main().
    """
    if not os.path.exists("data/vector_db/index.faiss"):
        st.error("No existing database found. Upload and process PDFs first.")
        return

    with st.spinner("Loading existing vector database..."):
        vector_store = FAISSVectorStore(embedding_dim=embedding_generator.embedding_dim)
        vector_store.load("data/vector_db")

    st.session_state.vector_store = vector_store
    st.session_state.rag_pipeline = ESGRAGPipeline(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )

    st.success(f"✅ Loaded {vector_store.index.ntotal} chunks from existing database")


# ---------------- UI Sections ----------------

def render_sidebar(embedding_generator):
    with st.sidebar:
        st.markdown("### 📄 Document Management")

        db_mode = st.radio(
            "Vector database",
            ["Load existing database", "Create new from uploads"],
            index=0
        )

        if db_mode == "Create new from uploads":
            uploaded_files = st.file_uploader(
                "Upload ESG Reports (PDF)",
                type=["pdf"],
                accept_multiple_files=True
            )

            if uploaded_files and st.button("Process Documents", type="primary"):
                process_uploaded_pdfs(uploaded_files, embedding_generator)

        else:
            if st.button("Load Existing Database", type="primary"):
                load_existing_database(embedding_generator)

        st.markdown('<hr class="esg-divider">', unsafe_allow_html=True)

        if st.session_state.vector_store:
            st.markdown(
                f'<span class="esg-status-pill">'
                f'{st.session_state.vector_store.index.ntotal} chunks indexed'
                f'</span>',
                unsafe_allow_html=True
            )
        else:
            st.caption("No documents loaded yet.")

        st.markdown('<hr class="esg-divider">', unsafe_allow_html=True)

        st.markdown("### ⚙️ Settings")

        st.session_state.k_value = st.slider(
            "Source chunks to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.k_value,
            help="Higher values retrieve more context but may dilute focus."
        )

        st.caption("Model · Gemini 2.5 Flash")
        st.caption("Embedding · all-MiniLM-L6-v2 (384-dim)")


def render_chat_interface():
    extractor = CitationExtractor()

    comparison_handler = ComparisonHandler(
        st.session_state.vector_store,
        st.session_state.embedding_generator
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_citation_chips(message.get("citations", []))

    if query := st.chat_input("Ask a question about the ESG reports..."):

        st.session_state.chat_history.append(
            {"role": "user", "content": query, "citations": []}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing reports..."):

                try:
                    if comparison_handler.is_comparison_question(query):
                        st.caption("🔍 Detected a comparison question — retrieving per-company context")
                        result = comparison_handler.query(query, k=st.session_state.k_value)
                    else:
                        result = st.session_state.rag_pipeline.query(
                            query, k=st.session_state.k_value
                        )

                    answer_text = result["answer"]
                    citations = extractor.extract_citations(answer_text)

                    st.markdown(answer_text)
                    render_citation_chips(citations)

                    with st.expander("📄 View retrieved source context"):
                        st.text(result["context"])

                except Exception as e:
                    answer_text = f"⚠️ Something went wrong while generating an answer: {e}"
                    citations = []
                    st.error(answer_text)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_text, "citations": citations}
        )


# ---------------- Main App ----------------

def main():
    load_css()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    render_hero()

    embedding_generator = load_embedding_generator()
    st.session_state.embedding_generator = embedding_generator

    if st.session_state.rag_pipeline is None and os.path.exists("data/vector_db/index.faiss"):
        load_existing_database(embedding_generator)

    render_sidebar(embedding_generator)

    if not st.session_state.rag_pipeline:
        st.markdown("👈 Upload ESG PDF reports in the sidebar, or load your existing database, to get started.")
        st.markdown("**Example questions once documents are loaded:**")
        for q in [
            "What is the carbon neutrality target?",
            "How much renewable energy is being used?",
            "What are the water conservation efforts?",
            "Compare Apple and Google's approach to emissions reduction.",
        ]:
            st.markdown(f'<div class="esg-example-card">{q}</div>', unsafe_allow_html=True)
        return

    render_chat_interface()


if __name__ == "__main__":
    main()