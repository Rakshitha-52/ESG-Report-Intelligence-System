import	sys
sys.path.append("src")
from	ingestion.pdf_loader	import	PDFLoader
loader	=	PDFLoader("data/pdfs")
documents	=	loader.load_all_pdfs()
#	Inspect	a	spread	of	pages
sample_indices	=	[0,	len(documents)	//	3,	len(documents)	//	2,	-1]
for	idx	in	sample_indices:
    doc	=	documents[idx]
    print(f"\n{'='*60}")
    print(f"Source:	{doc['metadata']['source']}	|	Page:{doc['metadata']['page']}")
    print(f"{'='*60}")
    print(doc["text"][:400])
    print("...")