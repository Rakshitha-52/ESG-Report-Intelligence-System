import	sys
sys.path.append("src")
from	ingestion.pdf_loader	import	PDFLoader
from	ingestion.text_preprocessor	import	TextPreprocessor
loader	=	PDFLoader("data/pdfs")
documents	=	loader.load_all_pdfs()
preprocessor	=	TextPreprocessor()
cleaned_docs	=	preprocessor.preprocess_documents(documents)

seen_sources	=	set()
for	raw,	clean	in	zip(documents,	cleaned_docs):
				source	=	raw["metadata"]["source"]
				if	source	not	in	seen_sources	and	raw["metadata"]["page"]	>	2:
								seen_sources.add(source)
								print(f"\n{'='*70}")
								print(f"SOURCE:	{source}")
								print(f"{'='*70}")
								print("---	BEFORE	---")
								print(raw["text"][:300])
								print("\n---	AFTER	---")
								print(clean["text"][:300])