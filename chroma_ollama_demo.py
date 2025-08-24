# chroma_ollama_demo.py
# Uses ChromaDB + Ollama embeddings

import numpy as np
import ollama
import chromadb
from chromadb.config import Settings

EMBED_MODEL = "nomic-embed-text"   # 2048-dim, local via Ollama
PERSIST_DIR = "./chroma_store"  # set to a folder path (e.g. "./chroma_store") to persist between runs

# --- Custom embedding function for ChromaDB ---
class OllamaEmbeddingFunction:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Chroma will pass a list[str]; return list of vectors (list[list[float]])
        vectors = []
        for t in input:
            r = ollama.embeddings(model=self.model, prompt=t)
            vectors.append(r["embedding"])
        return vectors

# --- Initialize ChromaDB client & collection ---
client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    persist_directory=PERSIST_DIR  # set a path to persist; None = in-memory
))

COLLECTION_NAME = "pricing_demo"

# Clean up any stale collection for idempotent runs (optional)
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=OllamaEmbeddingFunction(EMBED_MODEL),
    metadata={"hnsw:space": "cosine"}  # ensure cosine distance (recommended for embeddings)
)

# --- Step 1: Add documents (Chroma will call our embedder) ---
docs = [
    "Azure AI pricing updated on Aug 2025: $0.001 per token",
    "AWS AI pricing July 2025: $0.002 per token",
    "Google AI pricing May 2025: $0.0015 per token"
]
ids = [f"doc-{i}" for i in range(len(docs))]
metadatas = [{"source": "demo", "vendor": v.split()[0]} for v in docs]

collection.add(
    ids=ids,
    documents=docs,
    metadatas=metadatas
)

# --- Step 2/3: Query for the most relevant documents ---
query = "Azure AI pricing updates"
results = collection.query(
    query_texts=[query],
    n_results=1,
    include=["documents", "metadatas", "distances"]
)

best_doc = results["documents"][0][0]
best_dist = results["distances"][0][0]  # cosine distance (lower is better)
best_meta = results["metadatas"][0][0]

print(f"Query: {query!r}")
print(f"Best Match: {best_doc}")
print(f"Distance: {best_dist:.4f} (cosine distance; lower = closer)")
print(f"Metadata: {best_meta}")
