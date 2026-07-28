"""
Citation Extraction Module

Parses [Source: filename, Page X] citations out of LLM answers
so they can be displayed separately and verified in the UI.

Handles both formats Gemini may produce:
  - Separate brackets:  [Source: a.pdf, Page 1] [Source: b.pdf, Page 2]
  - Combined bracket:   [Source: a.pdf, Page 1; Source: b.pdf, Page 2]
"""

import re
from typing import List, Dict


class CitationExtractor:
    """Extract and deduplicate citations from a RAG answer."""

    def __init__(self):
        # No longer anchored to a literal closing "]" right after the page
        # number - this is what allows it to match citations that Gemini
        # packs together in a single bracket separated by ";".
        self.pattern = re.compile(r"Source:\s*([^,\]]+),\s*Page\s*(\d+)")

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
    
    combined_bracket_answer = (
        "Google announced the pursuit of a 100% renewable energy match "
        "[Source: google_esg_2025.pdf, Page 29; Source: google_esg_2026.pdf, Page 28]."
    )

    separate_bracket_answer = (
        "Apple has committed to carbon neutrality by 2030 "
        "[Source: apple_esg_2025.pdf, Page 12]. This includes a 75% reduction "
        "target [Source: apple_esg_2025.pdf, Page 15]."
    )

    extractor = CitationExtractor()

    print("Combined-bracket test:")
    print(extractor.extract_citations(combined_bracket_answer))

    print("\nSeparate-bracket test:")
    print(extractor.extract_citations(separate_bracket_answer))