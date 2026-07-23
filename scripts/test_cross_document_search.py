import sys

sys.path.append("src")

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore


def main():
    # Load embedding model
    generator = EmbeddingGenerator()

    # Load saved vector database
    vector_store = FAISSVectorStore(
        embedding_dim=generator.embedding_dim
    )
    vector_store.load("data/vector_db")

    # ---------------------------------------------------------
    # Test 1: Company-specific query
    # ---------------------------------------------------------
    query = "Apple carbon emissions reduction target"

    query_embedding = generator.embed_text(query)
    results = vector_store.search(query_embedding, k=5)

    print("=" * 70)
    print("TEST 1: Company-specific query")
    print(f"Query: {query}\n")

    for text, metadata, dist in results:
        print(
            f"{metadata['source']} "
            f"(Page {metadata['page']}) "
            f"- Distance: {dist:.4f}"
        )

    # ---------------------------------------------------------
    # Test 2: Generic ESG query
    # ---------------------------------------------------------
    query2 = "renewable energy usage percentage"

    query_embedding2 = generator.embed_text(query2)
    results2 = vector_store.search(query_embedding2, k=8)

    print("\n" + "=" * 70)
    print("TEST 2: Generic query across companies")
    print(f"Query: {query2}\n")

    sources_seen = set()

    for text, metadata, dist in results2:
        sources_seen.add(metadata["source"])

        print(
            f"{metadata['source']} "
            f"(Page {metadata['page']}) "
            f"- Distance: {dist:.4f}"
        )

    print(f"\nUnique sources in top 8 results: {len(sources_seen)}")
    print(
        "Expectation: Generic queries should ideally retrieve "
        "results from multiple companies."
    )


