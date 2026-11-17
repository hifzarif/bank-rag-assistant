import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_PATH = "../data/processed/faiss.index"

_model, _index = None, None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model

def build_index():
    chunks = json.load(open("../data/processed/chunks.json"))
    texts = [c["text"] for c in chunks]
    emb = get_model().encode(texts, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine, since normalized
    index.add(np.asarray(emb, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    print("Indexed", index.ntotal, "chunks, dimension", emb.shape[1])

def search(query, k=5):
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    q = get_model().encode([QUERY_PREFIX + query], normalize_embeddings=True)
    scores, ids = _index.search(np.asarray(q, dtype="float32"), k)
    return list(zip(ids[0].tolist(), scores[0].tolist()))

if __name__ == "__main__":
    build_index()