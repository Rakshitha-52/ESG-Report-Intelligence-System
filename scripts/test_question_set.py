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

edge_case_questions = [
    "What is the CEO's personal opinion on cryptocurrency?",
    "What is the company's marketing budget for next year?",
    "How many pets does the sustainability director own?",
]

for q in edge_case_questions:
    result = rag.query(q, k=5)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    time.sleep(1)

comparison_questions = [
    "Compare Apple and Google's approach to carbon neutrality.",
    "How did Microsoft's renewable energy usage change between the two reports?",
    "Which company has a more ambitious net-zero target, Reliance or Tata?",
]

for q in comparison_questions:
    result = rag.query(q, k=8)  # more chunks for multi-entity questions
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    time.sleep(1)