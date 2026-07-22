"""
Text Cleaning and Preprocessing Module

Cleans raw PDF-extracted text before chunking:
- Removes page numbers
- Removes repeated headers/footers
- Normalizes whitespace
- Fixes broken line wraps
"""

import re
from typing import List, Dict


class TextPreprocessor:
    """Clean and normalize extracted PDF text."""

    def __init__(self, known_headers: List[str] = None):
        """
        Args:
            known_headers: Optional list of exact header/footer strings
                that repeat across pages of a specific report
                (e.g., "Google 2025 Environmental Report").

                Pass per-document if you notice repeated boilerplate
                during your manual review of the report.
        """
        self.known_headers = known_headers or []

        self.patterns = {
            # A line that contains only digits (page number)
            "page_numbers": re.compile(
                r"^\s*\d{1,4}\s*$",
                re.MULTILINE
            ),

            # Common copyright/legal footer lines
            "copyright_footer": re.compile(
                r"^(©.*|All rights reserved.*|Confidential.*)$",
                re.MULTILINE | re.IGNORECASE
            ),

            # Multiple spaces/tabs → single space
            "extra_whitespace": re.compile(r"[ \t]+"),

            # 3+ blank lines → 2 blank lines
            "extra_blank_lines": re.compile(r"\n{3,}"),

            # Fix hyphenated line breaks:
            # emis-\nsions → emissions
            "hyphen_linebreak": re.compile(r"(\w+)-\n(\w+)"),
        }

    def remove_known_headers(self, text: str) -> str:
        """
        Remove exact repeated header/footer strings
        for this document.
        """
        for header in self.known_headers:
            text = text.replace(header, "")

        return text

    def clean_text(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to one page's text.

        Args:
            text: Raw text extracted from PDF

        Returns:
            Cleaned text
        """

        # 1. Remove known repeated headers/footers
        text = self.remove_known_headers(text)

        # 2. Fix hyphenated line-wrap breaks
        text = self.patterns["hyphen_linebreak"].sub(r"\1\2", text)

        # 3. Remove standalone page numbers
        text = self.patterns["page_numbers"].sub("", text)

        # 4. Remove copyright/legal footers
        text = self.patterns["copyright_footer"].sub("", text)

        # 5. Normalize whitespace
        text = self.patterns["extra_whitespace"].sub(" ", text)
        text = self.patterns["extra_blank_lines"].sub("\n\n", text)

        # 6. Remove leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_documents(
        self,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        Clean a list of page-level documents.

        Args:
            documents: List of dictionaries:
                {
                    "text": "...",
                    "metadata": {...}
                }

        Returns:
            Cleaned documents with empty pages removed
        """

        cleaned_docs = []

        for doc in documents:
            cleaned_text = self.clean_text(doc["text"])

            # Skip empty/near-empty pages
            if cleaned_text and len(cleaned_text) > 30:
                cleaned_docs.append(
                    {
                        "text": cleaned_text,
                        "metadata": doc["metadata"]
                    }
                )

        print(
            f"Cleaned {len(cleaned_docs)} pages "
            f"(dropped {len(documents) - len(cleaned_docs)} "
            f"empty/near-empty pages)"
        )

        return cleaned_docs


if __name__ == "__main__":
    print("TextPreprocessor module loaded successfully.")