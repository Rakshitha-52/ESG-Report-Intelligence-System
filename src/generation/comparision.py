"""
Multi-Document Comparison Module

Handles "compare X and Y" style questions by retrieving
context per entity (filtered by source metadata) rather than
relying on a single combined similarity search, which can
under-represent one side of the comparison.
"""

from typing import Dict, List
import google.generativeai as genai


# Maps a spoken company name to the filename fragments that identify
# its reports in your vector store's metadata["source"] field.
# Update this if your actual filenames differ.
COMPANY_SOURCE_MAP = {
    "apple": [
        "apple_esg_2025.pdf",
        "apple_esg_2026.pdf",
    ],
    "google": [
        "google_esg_2025.pdf",
        "google_esg_2026.pdf",
    ],
    "microsoft": [
        "microsoft_esg_2025.pdf",
        "microsoft_esg_2026.pdf",
    ],
    "reliance": [
        "reliance_esg_2025.pdf",
    ],
    "tata": [
        "tata_esg_2025.pdf",
    ],
}


class ComparisonHandler:
    """Detects and answers multi-entity comparison questions."""

    COMPARISON_PROMPT = """
You are an ESG analyst comparing companies.

Use ONLY the labeled context sections below—each section belongs to a specific company.

{context}

QUESTION:
{question}

RULES:
1. Address each company separately, then give a direct comparison.
2. Use only information from that company's labeled section for claims about it.
3. Cite every factual claim using this exact format:
   [Source: filename, Page X]
   Use ONE bracket per citation—do not combine multiple sources in one bracket.
4. If one company's section lacks information to answer part of the question,
   say so explicitly for that company rather than guessing.
5. Be specific—include exact numbers, percentages, and dates when available.

ANSWER:
"""

    def __init__(
        self,
        vector_store,
        embedding_generator,
        model_name: str = "gemini-1.5-flash",
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.model = genai.GenerativeModel(model_name)

    def detect_companies(self, question: str) -> List[str]:
        """Find which known companies are mentioned in the question."""

        question_lower = question.lower()

        return [
            company
            for company in COMPANY_SOURCE_MAP
            if company in question_lower
        ]

    def is_comparison_question(self, question: str) -> bool:
        """
        A question is treated as a comparison if:
        - Two or more known companies are mentioned, OR
        - Comparison language is used alongside at least one company.
        """

        companies = self.detect_companies(question)

        comparison_words = [
            "compare",
            "vs",
            "versus",
            "difference between",
            "which company",
            "better than",
        ]

        has_comparison_word = any(
            word in question.lower()
            for word in comparison_words
        )

        return (
            len(companies) >= 2
            or (len(companies) >= 1 and has_comparison_word)
        )

    def retrieve_per_entity(
        self,
        question: str,
        companies: List[str],
        k: int = 4,
    ) -> Dict[str, str]:
        """
        Retrieve top-k chunks separately for each company,
        filtered to only that company's source files.
        """

        query_embedding = self.embedding_generator.embed_text(question)

        per_company_context = {}

        for company in companies:
            valid_sources = set(COMPANY_SOURCE_MAP.get(company, []))

            # Over-fetch, then filter by source, then trim to k.
            # This is simple and explainable for a small dataset.
            # In production, FAISS metadata filtering would be preferable.
            raw_results = self.vector_store.search(
                query_embedding,
                k=k * 5,
            )

            filtered = [
                (text, meta, dist)
                for text, meta, dist in raw_results
                if meta.get("source") in valid_sources
            ][:k]

            context_parts = [
                f"[Source: {meta['source']}, Page {meta['page']}]\n{text}"
                for text, meta, dist in filtered
            ]

            per_company_context[company] = (
                "\n\n".join(context_parts)
                if context_parts
                else "(No relevant context found for this company.)"
            )

        return per_company_context

    def query(
        self,
        question: str,
        k: int = 4,
    ) -> Dict:
        """Answer a comparison question end-to-end."""

        companies = self.detect_companies(question)

        per_company_context = self.retrieve_per_entity(
            question,
            companies,
            k=k,
        )

        labeled_context = "\n\n===\n\n".join(
            f"### {company.upper()}\n{context}"
            for company, context in per_company_context.items()
        )

        prompt = self.COMPARISON_PROMPT.format(
            context=labeled_context,
            question=question,
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1
            ),
        )

        return {
            "question": question,
            "answer": response.text,
            "context": labeled_context,
            "companies_detected": companies,
        }


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv

    sys.path.append("src")

    from retrieval.embeddings import EmbeddingGenerator
    from retrieval.vector_store_faiss import FAISSVectorStore

    load_dotenv()

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    generator = EmbeddingGenerator()

    vector_store = FAISSVectorStore(
        embedding_dim=generator.embedding_dim
    )
    vector_store.load("data/vector_db")

    handler = ComparisonHandler(
        vector_store,
        generator,
    )

    q = "Compare Apple and Google's approach to carbon neutrality."

    print(
        f"Is comparison question: "
        f"{handler.is_comparison_question(q)}"
    )

    print(
        f"Companies detected: "
        f"{handler.detect_companies(q)}\n"
    )

    result = handler.query(q)

    print(result["answer"])