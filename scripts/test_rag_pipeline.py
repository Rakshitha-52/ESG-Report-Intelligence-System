"""
End-to-end RAG pipeline test: retrieval -> Gemini -> citations.
Run this after Steps 5 and 6 are both complete.
"""

import sys
sys.path.append("src")

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore
from generation.rag_chain import ESGRAGPipeline
from generation.citation_extractor import CitationExtractor


def main():
    generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore(embedding_dim=generator.embedding_dim)
    vector_store.load("data/vector_db")

    rag = ESGRAGPipeline(vector_store=vector_store, embedding_generator=generator)
    extractor = CitationExtractor()

    question = "What is the carbon neutrality target?"
    result = rag.query(question, k=5)

    formatted_answer = extractor.format_with_source_list(result["answer"])

    print(f"Question: {question}\n")
    print(formatted_answer)


if __name__ == "__main__":
    main()