"""
RAG Pipeline Module (Gemini-powered)

Combines Day 3's retrieval system with Google's Gemini API to
produce grounded, cited answers to ESG questions - at zero cost
on the free tier.
"""

import os
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class ESGRAGPipeline:
    """Retrieval-Augmented Generation pipeline for ESG report Q&A."""

    SYSTEM_PROMPT = """You are an ESG analyst. Answer using ONLY the context below.

CONTEXT:
{context}

QUESTION: {question}

RULES:
1. Answer only using information found in the context above.
2. If the answer is not in the context, respond exactly with:
   "I don't have enough information in the provided reports to answer this question."
3.  Cite every factual claim using this exact format: [Source: filename, Page X]
   Use ONE bracket per citation. If a fact appears in multiple documents,
   write separate brackets like this: [Source: file1.pdf, Page 1] [Source: file2.pdf, Page 2]
   Do NOT combine multiple sources into a single bracket with a semicolon.
4. Do not use any outside knowledge about the company, even if you know it.
5. Be specific - include exact numbers, percentages, and dates when available.
6. Do not repeat the same fact twice in different words. State each point once.

ANSWER:"""

    def __init__(self, vector_store, embedding_generator,
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.1):
        """
        Args:
            vector_store: A loaded FAISSVectorStore instance (Day 3)
            embedding_generator: An EmbeddingGenerator instance (Day 3)
            model_name: Gemini model identifier - 'gemini-1.5-flash'
                is fast, free-tier friendly, and sufficient for this task
            temperature: Near-zero for grounded, deterministic answers
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.temperature = temperature
        self.model = genai.GenerativeModel(model_name)

    

    def retrieve_context(self, question: str, k: int = 5) -> str:
        """
        Retrieve the top-k relevant chunks and format them into
        a labeled context block Gemini can cite from directly.
        """
        query_embedding = self.embedding_generator.embed_text(question)
        results = self.vector_store.search(query_embedding, k=k)


        if	not	results:
            return	"(No	documents	available	to	search.)"

        context_parts = []
        for idx, (text, metadata, distance) in enumerate(results, 1):
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")
            context_parts.append(
                f"[Document {idx}] [Source: {source}, Page {page}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def query(self, question: str, k: int = 5) -> Dict:
        """
        Answer a question end-to-end: retrieve -> prompt -> generate.

        Returns:
            Dict with the question, generated answer, retrieved
            context (for debugging/display), and k used.
        """
        context = self.retrieve_context(question, k=k)
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )

        answer = response.text

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "k": k
        }


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from retrieval.embeddings import EmbeddingGenerator
    from retrieval.vector_store_faiss import FAISSVectorStore

    generator = EmbeddingGenerator()
    vector_store = FAISSVectorStore(embedding_dim=generator.embedding_dim)
    vector_store.load("data/vector_db")

    rag = ESGRAGPipeline(vector_store=vector_store, embedding_generator=generator)

    result = rag.query("What is the carbon neutrality target?", k=5)
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")