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

# ---------------- Main App ----------------

def main():
    st.title("📄 ESG Report Intelligence System")
    st.caption(
        "Ask questions about ESG and sustainability reports, "
        "with cited answers."
    )

if	__name__	==	"__main__":
				main()