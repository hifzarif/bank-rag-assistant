import json
from index import search
from baseline import tfidf_search

chunks = {c["id"]: c for c in json.load(open("../data/processed/chunks.json"))}

q = input("Question: ")
for name, fn in [("EMBEDDINGS", search), ("TF-IDF", tfidf_search)]:
    print(f"\n=== {name} ===")
    for cid, score in fn(q, 3):
        c = chunks[cid]
        print(f"[{score:.2f}] {c['doc']} p.{c['page']}: {c['text'][:200]}...")