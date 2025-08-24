# Requires: pip install ollama numpy
# And ensure the Ollama daemon is running: `ollama serve`
# First time only: `ollama pull nomic-embed-text`

import numpy as np
import ollama

sentences = [
    "I love eating apples",
    "Oranges are my favorite",
    "The king rules the kingdom"
]

def embed_texts_ollama(texts, model="nomic-embed-text"):
    """
    Returns a 2D numpy array of embeddings (n_texts x dim)
    """
    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=model, prompt=t)
        vectors.append(resp["embedding"])
    return np.array(vectors, dtype=np.float32)

# 1) Get embeddings
embeddings = embed_texts_ollama(sentences, model="nomic-embed-text")

# 2) Unit-normalize for true cosine via dot product
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_unit = embeddings / np.clip(norms, 1e-12, None)

# 3) Cosine similarity matrix = dot product of unit vectors
similarities = embeddings_unit @ embeddings_unit.T

# 4) Clean tiny FP noise for neat printing
similarities = np.clip(similarities, -1.0, 1.0)

print(np.round(similarities, 3))
