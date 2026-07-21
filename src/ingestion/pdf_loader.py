import os
from pathlib import Path
from typing import List, Dict

import PyPDF2
import pdfplumber


class PDFLoader:

    def __init__(self, pdf_dir: str):
        self.pdf_dir = Path(pdf_dir)

        if not self.pdf_dir.exists():
            raise ValueError(
                f"PDF directory not found: {self.pdf_dir}"
            )

    def load_pdf_pypdf2(self, file_path: str) -> List[Dict]:

        documents = []

        with open(file_path, "rb") as file:

            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):

                text = page.extract_text()

                if not text or not text.strip():
                    continue

                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1,
                            "total_pages": len(pdf_reader.pages),
                            "method": "pypdf2",
                        },
                    }
                )

        return documents

    def load_pdf_pdfplumber(self, file_path: str) -> List[Dict]:

        documents = []

        with pdfplumber.open(file_path) as pdf:

            for page_num, page in enumerate(pdf.pages):

                text = page.extract_text()

                if not text or not text.strip():
                    continue

                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1,
                            "total_pages": len(pdf.pages),
                            "method": "pdfplumber",
                        },
                    }
                )

        return documents

    def load_single_pdf(
        self,
        file_path: str,
        method: str = "pdfplumber",
    ) -> List[Dict]:

        if method == "pypdf2":
            docs = self.load_pdf_pypdf2(file_path)
        else:
            docs = self.load_pdf_pdfplumber(file_path)

        if not docs:

            print(
                f"⚠ No text extracted using {method}. "
                f"Trying fallback..."
            )

            if method == "pdfplumber":
                docs = self.load_pdf_pypdf2(file_path)
            else:
                docs = self.load_pdf_pdfplumber(file_path)

        return docs

    def load_all_pdfs(
        self,
        method: str = "pdfplumber",
    ) -> List[Dict]:

        all_documents = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(
                f"No PDF files found in {self.pdf_dir}"
            )

        for pdf_file in pdf_files:

            print(f"\nLoading {pdf_file.name}")

            docs = self.load_single_pdf(
                str(pdf_file),
                method=method,
            )

            print(
                f"✓ Extracted {len(docs)} pages"
            )

            all_documents.extend(docs)

        print(
            f"\nTotal pages extracted: "
            f"{len(all_documents)}"
        )

        return all_documents


if __name__ == "__main__":

    # ESG project root
    project_root = Path(__file__).resolve().parents[2]

    pdf_dir = project_root / "data" / "pdfs"

    print("Project Root:", project_root)
    print("PDF Directory:", pdf_dir)

    loader = PDFLoader(str(pdf_dir))

    documents = loader.load_all_pdfs()

    if documents:

        print("\nSample Text:\n")

        print(
            documents[0]["text"][:500]
        )

        print(
            "\nMetadata:",
            documents[0]["metadata"]
        )