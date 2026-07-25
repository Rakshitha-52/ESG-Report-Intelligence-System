"""
ESG Report Intelligence System - Streamlit Application

Main entry point. Provides PDF upload, chat-based Q&A,
and cited answers powered by the RAG pipeline.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append("src")

from ingestion.pdf_loader import PDFLoader
from ingestion.text_preprocessor import TextPreprocessor
from ingestion.chunker import DocumentChunker
from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore
from generation.rag_chain import ESGRAGPipeline
from generation.citation_extractor import CitationExtractor

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="ESG Report Intelligence",
    page_icon="📄",
    layout="wide"
)

# ---------------- Session State ----------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Cached Resources ----------------

@st.cache_resource
def load_embedding_generator():
    return EmbeddingGenerator()

def process_uploaded_pdfs(uploaded_files, embedding_generator):
    """Run the full Day 1-3 pipeline on newly uploaded PDFs."""

    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded PDFs
    for uploaded_file in uploaded_files:
        with open(pdf_dir / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Extract text
    with st.spinner("Extracting text from PDFs..."):
        loader = PDFLoader(str(pdf_dir))
        documents = loader.load_all_pdfs()

    # Clean text
    with st.spinner("Cleaning text..."):
        preprocessor = TextPreprocessor()
        cleaned_docs = preprocessor.preprocess_documents(documents)

    # Chunk documents
    with st.spinner("Chunking documents..."):
        chunker = DocumentChunker(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = chunker.chunk_documents(cleaned_docs)

    # Generate embeddings
    with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        embeddings = embedding_generator.embed_batch(
            texts,
            show_progress=False
        )

    # Build vector database
    with st.spinner("Building vector database..."):
        vector_store = FAISSVectorStore(
            embedding_dim=embedding_generator.embedding_dim
        )

        vector_store.add_documents(
            texts,
            embeddings,
            metadatas
        )

        vector_store.save("data/vector_db")

    # Store in session state
    st.session_state.vector_store = vector_store

    st.session_state.rag_pipeline = ESGRAGPipeline(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )

    st.success(
        f"✅ Processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks"
    )

# ---------------- Main App ----------------

def main():
    st.title("📄 ESG Report Intelligence System")
    st.caption(
        "Ask questions about ESG and sustainability reports, "
        "with cited answers."
    )

    # ---------------- Sidebar ----------------

    embedding_generator = load_embedding_generator()

    with st.sidebar:
        st.header("📂 Document Management")

        uploaded_files = st.file_uploader(
            "Upload ESG Reports (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button(
            "Process Documents",
            type="primary"
        ):
            process_uploaded_pdfs(
                uploaded_files,
                embedding_generator
            )

        st.divider()

        if st.session_state.vector_store:
            st.success(
                f"✅ {st.session_state.vector_store.index.ntotal} chunks indexed"
            )
        else:
            st.info(
                "No documents loaded yet. Upload PDFs or load the existing database below."
            )

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

def render_chat_interface():
    """Render chat history and handle new questions."""

    extractor = CitationExtractor()

    # Replay chat history on every rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about the ESG reports..."):

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing reports..."):

                result = st.session_state.rag_pipeline.query(
                    query,
                    k=5
                )

                formatted_answer = extractor.format_with_source_list(
                    result["answer"]
                )

                st.markdown(formatted_answer)

                with st.expander("📖 View retrieved source context"):
                    st.text(result["context"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": formatted_answer
            }
        )

if	__name__	==	"__main__":
	main()