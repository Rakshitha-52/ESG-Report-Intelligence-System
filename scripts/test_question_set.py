import sys
import time
sys.path.append("src")

from retrieval.embeddings import EmbeddingGenerator
from retrieval.vector_store_faiss import FAISSVectorStore
from generation.rag_chain import ESGRAGPipeline
from generation.citation_extractor import CitationExtractor

generator = EmbeddingGenerator()
vector_store = FAISSVectorStore(embedding_dim=generator.embedding_dim)
vector_store.load("data/vector_db")

rag = ESGRAGPipeline(vector_store=vector_store, embedding_generator=generator)
extractor = CitationExtractor()

test_questions = [
    "What is Google's renewable energy target?",
    "What are Microsoft's water conservation efforts?",
    "What is Tata's approach to waste management?",
    "What are Reliance's sustainability initiatives?",
    "What is the scope 1 and scope 2 emissions breakdown for Apple?",
    "What percentage of energy comes from renewable sources at Google?",
]

for q in test_questions:
    result = rag.query(q, k=5)
    formatted = extractor.format_with_source_list(result["answer"])
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"{'='*70}")
    print(formatted)
    time.sleep(1)  # small pause - stays comfortably within free-tier rate limits