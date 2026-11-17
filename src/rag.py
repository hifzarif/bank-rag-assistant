import json, ollama
from index import search

chunks = {c["id"]: c for c in json.load(open("../data/processed/chunks.json"))}
LLM = "llama3.2:3b"

SYSTEM = """You are a banking document assistant.
Answer ONLY using the numbered sources provided.
Cite sources inline like [1] or [2][3] after each claim.
If the sources do not contain the answer, say exactly: "I could not find this in the provided documents."
Never use outside knowledge. Be concise."""

def ask_llm(system, user, max_tokens=400):
    r = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
    )
    return r["message"]["content"].strip()

def answer(question, k=5):
    hits = search(question, k)
    context = "\n\n".join(
        f"[{n}] ({chunks[cid]['doc']}, p.{chunks[cid]['page']})\n{chunks[cid]['text']}"
        for n, (cid, _) in enumerate(hits, start=1)
    )
    text = ask_llm(SYSTEM, f"Sources:\n{context}\n\nQuestion: {question}", 500)
    return {"answer": text,
            "sources": [{"n": n, **chunks[cid], "score": s}
                        for n, (cid, s) in enumerate(hits, start=1)]}

if __name__ == "__main__":
    q = input("Question: ")
    res = answer(q)
    print("\nANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['doc']} p.{s['page']}")