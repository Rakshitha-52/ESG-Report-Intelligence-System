import sys

sys.path.append("src")

from ingestion.pdf_loader import PDFLoader
from ingestion.text_preprocessor import TextPreprocessor

loader = PDFLoader("data/pdfs")
documents = loader.load_all_pdfs()

preprocessor = TextPreprocessor()
cleaned_docs = preprocessor.preprocess_documents(documents)

print("\n--- Sample Cleaned Page ---")
print(cleaned_docs[0]["text"][:500])