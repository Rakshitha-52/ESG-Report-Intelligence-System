"""
FAISS Vector Store Module

Stores chunk embeddings and supports fast similarity search,
with persistence to disk so you don't have to re-embed
every time you restart your work.
"""

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np


class FAISSVectorStore:
    """FAISS-based vector database for ESG report chunks."""

    def __init__(self, embedding_dim: int = 384):
        """
        Args:
            embedding_dim: Must match your embedding model's output
                           dimension (384 for all-MiniLM-L6-v2).
        """
        self.embedding_dim = embedding_dim

        # Create a FAISS index using L2 (Euclidean) distance
        self.index = faiss.IndexFlatL2(embedding_dim)

        # FAISS stores only vectors.
        # We store the corresponding text and metadata separately.
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
    ):
        """
        Add a batch of document chunks along with their embeddings and metadata.
        """

        self.index.add(embeddings.astype("float32"))
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        print(f"Added {len(texts)} chunks.")
        print(f"Total vectors in index: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Find the k most similar chunks.

        Returns:
            List of (chunk_text, metadata, distance)
        """

        query_vector = query_embedding.reshape(1, -1).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.documents):
                results.append(
                    (
                        self.documents[idx],
                        self.metadatas[idx],
                        float(dist),
                    )
                )

        return results

    def save(self, save_dir: str = "data/vector_db"):
        """
        Save the FAISS index and metadata to disk.
        """

        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(save_dir, "index.faiss"),
        )

        with open(os.path.join(save_dir, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)

        print(
            f"Saved vector database to {save_dir} "
            f"({self.index.ntotal} chunks)"
        )

    def load(self, load_dir: str = "data/vector_db"):
        """
        Load a previously saved vector database.
        """

        self.index = faiss.read_index(
            os.path.join(load_dir, "index.faiss")
        )

        with open(os.path.join(load_dir, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

        with open(os.path.join(load_dir, "metadata.pkl"), "rb") as f:
            self.metadatas = pickle.load(f)

        print(
            f"Loaded vector database from {load_dir} "
            f"({len(self.documents)} chunks)"
        )