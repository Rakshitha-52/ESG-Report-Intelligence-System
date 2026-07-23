from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings using a Sentence-BERT model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace sentence-transformers model identifier.
                        Default is a fast, free, 384-dimension model.
        """
        print(f"Loading embedding model: {model_name} (first run downloads ~80MB)")

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single piece of text (e.g., a user's query)."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed many chunks of text at once.
        Used when building the vector database.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        print(f"Generated {len(embeddings)} embeddings")
        return embeddings


if __name__ == "__main__":
    texts = [
        "Carbon emissions reduced by 30% in 2024",
        "Company cut greenhouse gas output significantly",
        "Employee satisfaction survey results improved",
    ]

    generator = EmbeddingGenerator()
    embeddings = generator.embed_batch(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")

    # Manual cosine similarity check
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    sim_related = cosine_similarity(embeddings[0], embeddings[1])
    sim_unrelated = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\nSimilarity (related sentences):   {sim_related:.4f}")
    print(f"Similarity (unrelated sentences): {sim_unrelated:.4f}")

    print("\nExpectation: the related pair's score should be noticeably higher.")