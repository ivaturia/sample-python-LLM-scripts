# rag_ollama_chroma.py
# Local RAG pipeline with an "I don't know" guardrail + fallback to retrieval

import ollama
import chromadb
import numpy as np
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
# ======== CONFIG ========
LLM_MODEL = "llama3.1:8b"          # or "mistral:latest"
EMBED_MODEL = "nomic-embed-text"    # local embedding model via Ollama
PERSIST_DIR = "./chroma_store"      # set to None for in-memory
COLLECTION_NAME = "ai_pricing_demo"

# RAG knobs
TOP_K = 3                 # how many chunks to retrieve
MAX_CHUNK_TOKENS = 200    # crude chunk size (characters-based for simplicity here)
MIN_SIMILARITY = 0.25     # if using cosine similarity (we'll request distances and convert)
STRICT_IDK = True         # the model must say exactly: "I don't know." when uncertain

# ======== Embedding function for Chroma ========
class OllamaEmbeddingFunction:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        vecs = []
        for t in input:
            r = ollama.embeddings(model=self.model, prompt=t)
            vecs.append(r["embedding"])
        return vecs

# ======== LLM helpers ========
SYSTEM_PROMPT_BASE = f"""
You are a careful AI assistant. If you are NOT reasonably confident in an answer OR the answer is not present in the user's provided context, you MUST respond with exactly:
I don't know.

- Do not guess.
- If you do answer, be concise and factual.
"""

def llm_answer(prompt: str, system: Optional[str] = None) -> str:
    """Ask the LLM directly with an 'I don't know' guardrail."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    res = ollama.chat(model=LLM_MODEL, messages=messages)
    return res["message"]["content"].strip()

def llm_with_context(question: str, context_docs: List[str]) -> str:
    """Ask the LLM but constrain it to the retrieved context."""
    context_block = "\n\n".join(f"- {d}" for d in context_docs)
    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly:
I don't know.

Context:
{context_block}

Question: {question}
"""
    return llm_answer(prompt, system=SYSTEM_PROMPT_BASE)

# ======== Chroma setup ========
def get_chroma_collection(name: str = COLLECTION_NAME, persist_dir: Optional[str] = PERSIST_DIR):
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory=persist_dir
    ))
    # re-use if exists, else create
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(
            name=name,
            embedding_function=OllamaEmbeddingFunction(EMBED_MODEL),
            metadata={"hnsw:space": "cosine"}  # cosine distance
        )
    return col

# ======== Simple chunking (character-based for demo) ========
def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars
    return chunks

# ======== Ingest documents into Chroma ========
def ingest_documents(collection, docs: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
    # Flatten into chunks
    ids = []
    chunk_docs = []
    chunk_metas = []

    for i, d in enumerate(docs):
        chunks = chunk_text(d, max_chars=MAX_CHUNK_TOKENS*4)  # heuristic conversion
        for j, ch in enumerate(chunks):
            ids.append(f"doc-{i}-chunk-{j}")
            chunk_docs.append(ch)
            md = {"doc_index": i, "chunk_index": j}
            if metadatas and i < len(metadatas):
                md.update(metadatas[i])
            chunk_metas.append(md)

    # Upsert
    collection.upsert(
        ids=ids,
        documents=chunk_docs,
        metadatas=chunk_metas
    )
    return len(ids)

# ======== Retrieval ========
def retrieve(collection, query: str, k: int = TOP_K) -> Dict[str, Any]:
    """
    Returns Chroma query results. Distances are cosine distances (lower=closer).
    We'll compute similarity = 1 - distance for readability.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    sims = [1.0 - float(d) for d in dists]  # cosine similarity
    return {"docs": docs, "metadatas": metas, "similarities": sims}

# ======== Orchestrator ========
def answer_question(question: str, collection) -> str:
    """
    1) Ask LLM directly (guardrailed). If it knows, return the answer unless it's "I don't know."
    2) If "I don't know.", do retrieval from Chroma and ask LLM again with context.
    3) If retrieval weak or LLM still "I don't know.", return "I don't know."
    """
    # Step 1: direct ask
    direct = llm_answer(
        prompt=f"Question: {question}\nProvide an accurate answer if you are reasonably sure.",
        system=SYSTEM_PROMPT_BASE
    )
    if direct.strip().lower() != "i don't know.":
        return direct  # model felt confident

    # Step 2: RAG fallback
    r = retrieve(collection, question, k=TOP_K)
    # Filter by similarity threshold
    paired = [(doc, sim) for doc, sim in zip(r["docs"], r["similarities"]) if sim >= MIN_SIMILARITY]

    if not paired:
        return "I don't know."

    context_docs = [d for d, _ in paired]
    rag_answer = llm_with_context(question, context_docs).strip()
    return rag_answer

# ======== Demo ========
if __name__ == "__main__":
    # Example small domain: pricing snippets
    docs = [
        "Azure AI pricing updated on Aug 2025: $0.001 per token",
        "AWS AI pricing July 2025: $0.002 per token",
        "Google AI pricing May 2025: $0.0015 per token"
    ]
    metadatas = [
        {"vendor": "Azure"},
        {"vendor": "AWS"},
        {"vendor": "Google"}
    ]

    collection = get_chroma_collection()

    # Optional: clear existing and re-ingest for a clean demo
    try:
        # If you want a clean state each run, drop and recreate:
        client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddingFunction(EMBED_MODEL),
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        pass

    num_chunks = ingest_documents(collection, docs, metadatas)
    print(f"Ingested {num_chunks} chunks.\n")

    # 1) A question likely unknown to general LLM but in our RAG docs:
    q1 = "What is the latest Azure AI pricing update?"
    print("Q1:", q1)
    a1 = answer_question(q1, collection)
    print("A1:", a1, "\n")

    # 2) A question the base LLM may answer by itself (no RAG needed):
    q2 = "What is the capital of France?"
    print("Q2:", q2)
    a2 = answer_question(q2, collection)
    print("A2:", a2, "\n")

    # 3) A question neither LLM nor RAG can answer (should return 'I don't know.'):
    q3 = "What is the square root of the CEO of Azure?"
    print("Q3:", q3)
    a3 = answer_question(q3, collection)
    print("A3:", a3, "\n")
