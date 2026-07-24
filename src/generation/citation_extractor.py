"""
Citation Extraction Module

Parses [Source: filename, Page X] citations out of LLM answers
so they can be displayed separately and verified in the UI (Day 5).
Works identically regardless of which LLM generated the answer.
"""

import re
from typing import List, Dict


class CitationExtractor:
    """Extract and deduplicate citations from a RAG answer."""

    def __init__(self):
        self.pattern = re.compile(r"\[Source:\s*([^,]+),\s*Page\s*(\d+)\]")

    def extract_citations(self, answer_text: str) -> List[Dict]:
        """Extract unique (source, page) citations from an answer."""
        matches = self.pattern.findall(answer_text)

        citations = []
        seen = set()
        for source, page in matches:
            key = (source.strip(), int(page))
            if key not in seen:
                citations.append({"source": key[0], "page": key[1]})
                seen.add(key)

        return citations

    def format_with_source_list(self, answer_text: str) -> str:
        """Append a clean, numbered source list below the raw answer."""
        citations = self.extract_citations(answer_text)

        if not citations:
            return answer_text

        formatted = answer_text + "\n\n**Sources:**\n"
        for idx, c in enumerate(citations, 1):
            formatted += f"{idx}. {c['source']}, Page {c['page']}\n"

        return formatted


if __name__ == "__main__":
    sample_answer = (
        "Apple has committed to carbon neutrality across its entire "
        "business by 2030 [Source: apple_esg_2025.pdf, Page 12]. "
        "This includes a 75% reduction target [Source: apple_esg_2025.pdf, Page 15]."
    )

    extractor = CitationExtractor()
    citations = extractor.extract_citations(sample_answer)
    print("Extracted citations:", citations)

    formatted = extractor.format_with_source_list(sample_answer)
    print(f"\n{formatted}")