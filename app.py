"""
ESG Report Intelligence System - Streamlit Application

Provides PDF upload, chat-based Q&A, and cited
answers powered by the RAG pipeline (FAISS + Gemini).

"""

import textwrap
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
    html = """
        <div class="esg-hero">
            <div class="esg-hero-eyebrow">Retrieval-Augmented ESG Analysis</div>
            <div class="esg-hero-title">ESG Report Intelligence</div>
            <div class="esg-hero-sub">
                Ask questions across sustainability reports. Every answer is grounded
                in the source text and cited by document and page.
            </div>
        </div>
    """
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


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
if "just_processed" not in st.session_state:
    st.session_state.just_processed = False


# ---------------- Cached Resources ----------------

@st.cache_resource
def load_embedding_generator():
    return EmbeddingGenerator()


# ---------------- Pipeline Functions ----------------

def process_uploaded_pdfs(uploaded_files, embedding_generator):
    """Process only newly uploaded PDFs and append them to the existing
    vector database, rather than rebuilding from every file on disk."""
    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    already_indexed = set()
    if st.session_state.vector_store is not None:
        already_indexed = {
            m.get("source") for m in st.session_state.vector_store.metadatas
        }

    valid_files = []
    skipped_duplicates = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.warning(f"Skipped {uploaded_file.name} — not a PDF file.")
            continue

        if uploaded_file.name in already_indexed:
            skipped_duplicates.append(uploaded_file.name)
            continue

        try:
            with open(pdf_dir / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            valid_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Could not save {uploaded_file.name}: {e}")

    if skipped_duplicates:
        st.info(f"Already indexed, skipped: {', '.join(skipped_duplicates)}")

    if not valid_files:
        st.error("No new valid PDF files to process.")
        return

    loader = PDFLoader(str(pdf_dir))
    documents = []

    try:
        with st.spinner(f"Extracting text from {len(valid_files)} new file(s)..."):
            for filename in valid_files:
                file_path = pdf_dir / filename
                docs = loader.load_single_pdf(str(file_path))
                documents.extend(docs)
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return

    if not documents:
        st.error(
            "No extractable text found in the new PDF(s). "
            "The file(s) may be scanned images without a text layer."
        )
        return

    with st.spinner("Cleaning text..."):
        preprocessor = TextPreprocessor()
        cleaned_docs = preprocessor.preprocess_documents(documents)

    with st.spinner("Chunking documents..."):
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(cleaned_docs)

    with st.spinner(f"Generating embeddings for {len(chunks)} new chunks..."):
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = embedding_generator.embed_batch(texts, show_progress=False)

    with st.spinner("Updating vector database..."):
        if st.session_state.vector_store is not None:
            vector_store = st.session_state.vector_store
        else:
            vector_store = FAISSVectorStore(embedding_dim=embedding_generator.embedding_dim)
            if os.path.exists("data/vector_db/index.faiss"):
                vector_store.load("data/vector_db")

        vector_store.add_documents(texts, embeddings, metadatas)
        vector_store.save("data/vector_db")

    st.session_state.vector_store = vector_store
    st.session_state.rag_pipeline = ESGRAGPipeline(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )

    st.session_state.chat_history = []
    st.session_state.just_processed = True

    st.toast(
        f"✅ Added {len(valid_files)} new PDF(s) ({len(chunks)} chunks) to the database",
        icon="🎉"
    )
    st.rerun()
    
def load_existing_database(embedding_generator):
    """Load a previously built vector database from disk, if present."""
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
    """Sidebar now only handles loading an existing database and settings.
    Uploading new documents lives in the main tab (render_upload_section)."""
    with st.sidebar:
        st.markdown("### 📂 Existing Database")

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


def render_upload_section(embedding_generator):
    """Upload + process new ESG PDFs, shown in the main tab as a
    styled card rather than tucked away in the sidebar."""
    st.markdown(
        """
        <div class="esg-upload-card">
            <div class="esg-upload-header">
                <span class="esg-upload-icon">📤</span>
                <span class="esg-upload-title">Add New ESG Reports</span>
            </div>
            <div class="esg-upload-sub">
                Upload one or more sustainability PDFs to add them to the searchable database.
                Longer reports may take a few minutes to process.
            </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload ESG Reports (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("📤 Process Documents", type="primary"):
        process_uploaded_pdfs(uploaded_files, embedding_generator)

    st.markdown("</div>", unsafe_allow_html=True)


def render_ready_banner():
    """One-time, prominent confirmation shown right after processing
    completes, so the user isn't left guessing whether it's safe to ask
    questions yet."""
    if st.session_state.just_processed:
        st.markdown(
            '<div class="esg-ready-banner">✅ '
            '<strong>Ready!</strong>&nbsp; Your documents are processed — '
            'ask a question in the chat below.</div>',
            unsafe_allow_html=True
        )
        st.session_state.just_processed = False


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
    #load_css()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    render_hero()

    embedding_generator = load_embedding_generator()
    st.session_state.embedding_generator = embedding_generator

    if st.session_state.rag_pipeline is None and os.path.exists("data/vector_db/index.faiss"):
        load_existing_database(embedding_generator)

    render_sidebar(embedding_generator)

    # Upload section always visible in the main tab, above the chat -
    # lets users add more documents even after a database is loaded.
    with st.expander("📤 Add more documents", expanded=not st.session_state.rag_pipeline):
        render_upload_section(embedding_generator)

    render_ready_banner()

    if not st.session_state.rag_pipeline:
        st.markdown("👈 Load your existing database from the sidebar, or upload PDFs above, to get started.")
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