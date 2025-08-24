# pip install ollama numpy
import numpy as np
import ollama

# ---- Data ----
sentences = [
    "I love eating apples",
    "Oranges are my favorite",
    "The king rules the kingdom"
]

EMBED_MODEL = "nomic-embed-text"

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _embed_texts(texts, model=EMBED_MODEL) -> np.ndarray:
    vecs = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)
        vecs.append(r["embedding"])
    return np.asarray(vecs, dtype=np.float32)

# ---- Model & Corpus Index ----
embeddings = _embed_texts(sentences)
embeddings = _normalize_rows(embeddings)  # cosine = dot
embeddings = embeddings.astype(np.float32)  # smaller & faster

# ---- Search Function ----
def search(query, top_k=3):
    """
    Returns a list of (score, sentence, index) sorted by descending similarity.
    Uses Ollama embeddings; cosine similarity via dot product on normalized vectors.
    """
    q_emb = _embed_texts([query])[0].astype(np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # cosine similarities via dot product (since normalized)
    sims = embeddings @ q_emb  # shape: (N,)

    # get top-k indices
    top_k = min(top_k, len(sentences))
    idxs = np.argpartition(-sims, range(top_k))[:top_k]
    idxs = idxs[np.argsort(-sims[idxs])]  # sort the top-k
    return [(float(sims[i]), sentences[i], int(i)) for i in idxs]

# ---- Pretty Printer ----
def show_results(query, top_k=3, min_score=0.3):
    results = search(query, top_k=top_k)
    print(f"\nQuery: {query!r}")
    kept = 0
    for rank, (score, text, i) in enumerate(results, 1):
        if score >= min_score:
            kept += 1
            print(f"{rank:>2}. {score:.3f}  →  {text}  [#{i}]")
    if kept == 0:
        print(f"(no results ≥ {min_score})")

# ---- Examples ----
show_results("I enjoy fruits", top_k=2, min_score=0.3)
show_results("royal authority", top_k=2, min_score=0.3)
