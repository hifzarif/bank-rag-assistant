import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

chunks = json.load(open("../data/processed/chunks.json"))
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
X = vec.fit_transform([c["text"] for c in chunks])

def tfidf_search(query, k=5):
    q = vec.transform([query])
    sims = linear_kernel(q, X).ravel()
    top = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]