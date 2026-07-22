import json

with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

sources = sorted(
    set(c["metadata"]["source"] for c in chunks)
)

for source in sources:
    source_chunks = [
        c for c in chunks
        if c["metadata"]["source"] == source
    ]

    mid_chunk = source_chunks[len(source_chunks) // 2]

    print(f"\n{'=' * 70}")
    print(f"{source} — {len(source_chunks)} total chunks")
    print(f"{'=' * 70}")

    print(mid_chunk["text"][:400])

    print(
        f"\n[chunk_id: "
        f"{mid_chunk['metadata']['chunk_id']}]"
    )