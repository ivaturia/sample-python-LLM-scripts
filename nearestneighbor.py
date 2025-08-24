# pip install ollama numpy
import numpy as np
import ollama

sentences = [
    "I love eating apples",
    "Oranges are my favorite",
    "The king rules the kingdom"
]

def embed_batch(texts, model="nomic-embed-text"):
    vecs = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)
        vecs.append(r["embedding"])
    return np.array(vecs, dtype=np.float32)

embeddings = embed_batch(sentences)

# L2-normalize so dot product == cosine similarity (like your MiniLM code)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
embeddings = embeddings / norms

# Similarity matrix
similarities = embeddings @ embeddings.T
similarities = np.clip(similarities, -1.0, 1.0)

# Nearest neighbor printout
for i, s in enumerate(sentences):
    sims = similarities[i].copy()
    sims[i] = -np.inf
    j = int(np.argmax(sims))
    print(f"Nearest to {s!r} → {sentences[j]!r} (score: {sims[j]:.3f})")
