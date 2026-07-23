"""
Builds the full vector database from Day 2's saved chunks.

Run this once your chunks.json is finalized.
Re-run any time you update your chunking/cleaning logic.
"""

import json
import sys

sys.path.append("src")

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore


def main():
    # 1. Load chunks from Day 2
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from Day 2")

    # Extract text and metadata
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # 2. Generate embeddings for every chunk
    generator = EmbeddingGenerator()

    embeddings = generator.embed_batch(
        texts,
        batch_size=32,
        show_progress=True,
    )

    # 3. Build the FAISS index
    vector_store = FAISSVectorStore(
        embedding_dim=generator.embedding_dim
    )

    vector_store.add_documents(
        texts,
        embeddings,
        metadatas,
    )

    # 4. Save everything
    vector_store.save("data/vector_db")

    print("\nVector database build complete.")
    print(f"Total chunks indexed: {vector_store.index.ntotal}")


if __name__ == "__main__":
    main()