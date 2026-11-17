import json, re
from ingest import load_pdfs

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_pages(pages, size=200, overlap=40, min_words=30):
    chunks, cid = [], 0
    for p in pages:
        words = clean(p["text"]).split()
        step = size - overlap
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            if len(piece) < min_words:
                continue
            chunks.append({"id": cid, "doc": p["doc"], "page": p["page"],
                           "text": " ".join(piece)})
            cid += 1
    return chunks

if __name__ == "__main__":
    chunks = chunk_pages(load_pdfs())
    json.dump(chunks, open("../data/processed/chunks.json", "w"))
    print(f"\nTOTAL: {len(chunks)} chunks")