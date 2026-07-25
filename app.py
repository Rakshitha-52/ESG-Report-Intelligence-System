"""
ESG Report Intelligence System - Streamlit Application

Main entry point. Provides PDF upload, chat-based Q&A, and cited
answers powered by the Day 3/4 RAG pipeline (FAISS + Gemini).

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append("src")

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
    """Load the Sentence-BERT model once per session, not per rerun."""
    return EmbeddingGenerator()


# ---------------- Pipeline Functions ----------------

def process_uploaded_pdfs(uploaded_files, embedding_generator):
    """Run the full Day 1-3 pipeline on newly uploaded PDFs."""
    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(pdf_dir / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting text from PDFs..."):
        loader = PDFLoader(str(pdf_dir))
        documents = loader.load_all_pdfs()

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

    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks")


def load_existing_database(embedding_generator):
    """Load the vector database already built on Day 3, if present."""
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
    """Render the sidebar: upload, existing DB loader, and settings."""
    with st.sidebar:
        st.header("📂 Document Management")

        uploaded_files = st.file_uploader(
            "Upload ESG Reports (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            process_uploaded_pdfs(uploaded_files, embedding_generator)

        if st.button("Load Existing Database"):
            load_existing_database(embedding_generator)

        st.divider()

        if st.session_state.vector_store:
            st.success(f"✅ {st.session_state.vector_store.index.ntotal} chunks indexed")
        else:
            st.info("No documents loaded yet. Upload PDFs or load the existing database above.")

        st.divider()
        st.header("⚙️ Settings")

        st.session_state.k_value = st.slider(
            "Number of source chunks to retrieve",
            min_value=1, max_value=10, value=st.session_state.k_value,
            help="Higher values retrieve more context but may dilute focus."
        )

        st.caption("Model: Gemini 1.5 Flash")
        st.caption("Embedding: all-MiniLM-L6-v2 (384-dim)")


def render_chat_interface():
    """Render chat history and handle new questions."""
    extractor = CitationExtractor()

    # Replay chat history on every rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about the ESG reports..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing reports..."):
                try:
                    result = st.session_state.rag_pipeline.query(
                        query, k=st.session_state.k_value
                    )
                    formatted_answer = extractor.format_with_source_list(result["answer"])
                    st.markdown(formatted_answer)

                    citations = extractor.extract_citations(result["answer"])
                    if citations:
                        st.caption(f"📚 {len(citations)} source(s) cited")
                    else:
                        st.caption("⚠️ No sources cited in this answer")

                    with st.expander("📖 View retrieved source context"):
                        st.text(result["context"])

                except Exception as e:
                    formatted_answer = f"⚠️ Something went wrong while generating an answer: {e}"
                    st.error(formatted_answer)

        st.session_state.chat_history.append({"role": "assistant", "content": formatted_answer})


# ---------------- Main App ----------------

def main():
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    st.title("📄 ESG Report Intelligence System")
    st.caption("Ask questions about ESG and sustainability reports, with cited answers.")

    embedding_generator = load_embedding_generator()

    # Auto-load existing database on first run, if present
    if st.session_state.rag_pipeline is None and os.path.exists("data/vector_db/index.faiss"):
        load_existing_database(embedding_generator)

    render_sidebar(embedding_generator)

    if not st.session_state.rag_pipeline:
        st.info(
            "📄 Upload ESG PDF reports in the sidebar, or load your existing database, to get started."
        )
        st.markdown("""
        ### Example questions you can ask once documents are loaded:

        - What is the carbon neutrality target?
        - How much renewable energy is being used?
        - What are the water conservation efforts?
        - Compare Apple and Google's approach to emissions reduction.
        """)
        return  # Nothing else to render until documents are loaded

    render_chat_interface()


if __name__ == "__main__":
    main()