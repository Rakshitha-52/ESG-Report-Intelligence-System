import sys
import json

sys.path.append("src")

from ingestion.pdf_loader import PDFLoader
from ingestion.text_preprocessor import TextPreprocessor
from ingestion.chunker import DocumentChunker


KNOWN_HEADERS_BY_SOURCE = {
    "apple_esg_2025.pdf": ["Environmental Progress Report"],
    "apple_esg_2026.pdf": ["Environmental Progress Report"],
    "google_esg_2025.pdf": ["Google 2025 Environmental Report"],
    "google_esg_2026.pdf": ["Google 2026 Environmental Report"],
    "microsoft_esg_2025.pdf": ["Microsoft 2025 Sustainability Report"],
    "microsoft_esg_2026.pdf": ["Microsoft 2026 Sustainability Report"],
    "reliance_esg_2025.pdf": [],
    "tata_esg_2025.pdf": [],
}


def main():
    loader = PDFLoader("data/pdfs")
    raw_documents = loader.load_all_pdfs()


    all_cleaned = []

    for source, headers in KNOWN_HEADERS_BY_SOURCE.items():
        doc_pages = [
            d for d in raw_documents
            if d["metadata"]["source"] == source
        ]

        if not doc_pages:
            continue

        preprocessor = TextPreprocessor(known_headers=headers)
        cleaned = preprocessor.preprocess_documents(doc_pages)

        all_cleaned.extend(cleaned)

        print(
            f"{source}: {len(doc_pages)} raw pages -> "
            f"{len(cleaned)} cleaned pages"
        )

    # Chunk everything together
    chunker = DocumentChunker(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = chunker.chunk_documents(all_cleaned)

    print("\n--- Overall Chunk Statistics ---")
    for key, value in chunker.get_chunk_stats(chunks).items():
        print(f"  {key}: {value}")

    print("\n--- Chunks Per Source ---")
    for source, count in chunker.get_chunks_per_source(chunks).items():
        print(f"  {source}: {count} chunks")

    # Save chunks to disk as JSON 
    output_path = "data/processed/chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()