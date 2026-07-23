"""
Manual semantic search testing.

Confirms that the vector database retrieves relevant chunks
before moving on to the RAG pipeline.
"""

import sys

sys.path.append("src")

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore


def main():
    # Load embedding model
    generator = EmbeddingGenerator()

    # Load saved FAISS vector database
    vector_store = FAISSVectorStore(
        embedding_dim=generator.embedding_dim
    )
    vector_store.load("data/vector_db")

    # Test queries
    test_queries = [
        "What is the carbon neutrality target?",
        "How much renewable energy is being used?",
        "What are the water conservation efforts?",
        "What is the net-zero emissions goal?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Generate embedding for the query
        query_embedding = generator.embed_text(query)

        # Search top-3 most similar chunks
        results = vector_store.search(query_embedding, k=3)

        for rank, (text, metadata, distance) in enumerate(results, start=1):
            print(f"\n[Rank {rank}] Distance: {distance:.4f}")
            print(
                f"Source: {metadata['source']}, "
                f"Page: {metadata['page']}"
            )
            print(f"Text: {text[:200]}...")


