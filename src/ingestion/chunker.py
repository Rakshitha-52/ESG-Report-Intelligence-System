from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Split cleaned documents into retrieval-ready chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split cleaned page-level documents into chunks.

        Args:
            documents: List of {"text": ..., "metadata": {...}} dicts,
                       already cleaned by TextPreprocessor

        Returns:
            List of chunk dicts, each with enriched metadata including
            a unique chunk_id used later for citation tracking.
        """

        all_chunks = []

        for doc in documents:
            chunks = self.splitter.split_text(doc["text"])

            source = doc["metadata"]["source"]
            page = doc["metadata"]["page"]

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **doc["metadata"],
                    "chunk_id": f"{source}_p{page}_c{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "chunks_on_page": len(chunks)
                }

                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        print(
            f"Created {len(all_chunks)} chunks from "
            f"{len(documents)} pages"
        )

        return all_chunks

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Compute basic statistics to sanity-check chunk quality."""

        lengths = [len(c["text"]) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths)
        }

    def get_chunks_per_source(self, chunks: List[Dict]) -> Dict:
        """
        Count chunks generated per source PDF — useful for spotting
        a document that produced suspiciously few/many chunks.
        """

        counts = {}

        for chunk in chunks:
            source = chunk["metadata"]["source"]
            counts[source] = counts.get(source, 0) + 1

        return counts


if __name__ == "__main__":
    import sys

    sys.path.append("src")

    from ingestion.pdf_loader import PDFLoader
    from ingestion.text_preprocessor import TextPreprocessor

    loader = PDFLoader("data/pdfs")
    documents = loader.load_all_pdfs()

    preprocessor = TextPreprocessor()
    cleaned_docs = preprocessor.preprocess_documents(documents)

    chunker = DocumentChunker(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = chunker.chunk_documents(cleaned_docs)

    print("\n--- Chunk Statistics ---")
    for key, value in chunker.get_chunk_stats(chunks).items():
        print(f"  {key}: {value}")

    print("\n--- Chunks Per Source ---")
    for source, count in chunker.get_chunks_per_source(chunks).items():
        print(f"  {source}: {count} chunks")

    print("\n--- Sample Chunk ---")
    print(chunks[0]["text"][:300])

    print(f"\nMetadata: {chunks[0]['metadata']}")