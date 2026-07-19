import re
import numpy as np
from sentence_transformers import CrossEncoder

# import shared state from ingest (loaded/built once there)
from ingest import model, chunks, collection

# reranker used only here → load it here
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --- similarity helpers ---
def dot(a, b):
    return np.sum(np.array(a) * np.array(b))

def cosine(a, b):
    return dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- dense retrieval (brute-force cosine) ---
def retrieve(query, docs, k=3):
    q = model.encode(query)
    d_vecs = [model.encode(d) for d in docs]
    scores = [cosine(q, d) for d in d_vecs]
    return sorted(zip(scores, docs), reverse=True)[:k]


# --- keyword / hybrid ---
def keyword_score(query, doc) -> int:
    d_words = set(re.findall(r"\w+", doc.lower()))
    q_words = set(re.findall(r"\w+", query.lower()))
    return len(q_words & d_words)

def normalize(scores: list) -> list:
    mn, mx = min(scores), max(scores)
    out = []
    for s in scores:
        if mx == mn:
            out.append(0)
            continue
        out.append((s - mn) / (mx - mn))
    return out

def hybrid_retrieve(query, docs, k=3, alpha=0.5):
    q = model.encode(query)
    cos_scores = [cosine(q, model.encode(d)) for d in docs]
    kw_scores = [keyword_score(query, d) for d in docs]
    cos_norm = normalize(cos_scores)
    kw_norm = normalize(kw_scores)
    final = [alpha * c + (1 - alpha) * k for c, k in zip(cos_norm, kw_norm)]
    return sorted(zip(final, docs), reverse=True)[:k]


# --- Chroma retrieval ---
def retrieve_chroma(query, k=3):
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return results["documents"][0]


# --- reranking (two-stage) ---
def rerank_retrieve(query, k=3, initial_k=10):
    candidates = retrieve_chroma(query, k=initial_k)
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    return sorted(zip(scores, candidates), reverse=True)[:k]