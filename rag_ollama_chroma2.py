# rag_ollama_chroma2.py
# Local RAG pipeline with an "I don't know" guardrail + fallback to retrieval
# Shows which path answered (LLM vs RAG vs IDK) and prints retrieved context.

import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

import ollama
import chromadb
import numpy as np
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ---- ChromaDB + Ollama Embedding Setup ----
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "ai_pricing_demo"
PERSIST_DIR = "./chroma_store"

class OllamaEmbeddingFunction:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        vecs = []
        for t in input:
            r = ollama.embeddings(model=self.model, prompt=t)
            vecs.append(r["embedding"])
        return vecs

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

MAX_CHUNK_TOKENS = 200    # crude chunk size (characters-based for simplicity here)

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

# ---- LLM ----
LLM_MODEL = "llama3.1:8b"  # or "mistral:latest"

SYSTEM_PROMPT_BASE = """
You are a strict assistant.

RULES:
- If you are NOT 100% certain or the information is not in the provided context,
  you MUST answer exactly: I don't know.
- Do not provide general advice, links, guesses, or background info.
- Never try to be helpful when unsure.
- If you know, respond concisely and factually.

Example:
Q: What is the square root of the CEO of Azure?
A: I don't know.
""".strip()

def enforce_idk(text: str) -> str:
    lowered = text.lower()
    if "i don't know" in lowered or "i dont know" in lowered:
        return "I don't know."
    # if it rambles but didn’t follow rule, treat as IDK
    if "http" in lowered or "cannot provide" in lowered or "not aware" in lowered:
        return "I don't know."
    return text.strip()

def llm_answer(prompt: str, system: Optional[str] = SYSTEM_PROMPT_BASE) -> str:
    """Ask the LLM directly with an 'I don't know' guardrail."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    res = ollama.chat(model=LLM_MODEL, messages=messages)
    return enforce_idk(res["message"]["content"])

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

# ---- Retrieval ----
TOP_K = 3                   # how many chunks to retrieve
MIN_SIMILARITY = 0.25       # cosine similarity threshold for retrieval (1 - distance)

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

# ---- Result container ----
@dataclass
class AnswerResult:
    text: str
    source: str                     # "LLM", "RAG", or "IDK"
    retrieved: Optional[List[str]] = None
    similarities: Optional[List[float]] = None

# ---- Orchestrator ----
def answer_question(question: str, collection, prefer_direct: bool = True) -> AnswerResult:
    """
    prefer_direct=True:
      1) Ask LLM directly (guardrailed to say "I don't know." if unsure)
      2) If "I don't know.", do RAG (Chroma) and ask with context
      3) If retrieval weak or still unknown, return "I don't know." (source="IDK")

    prefer_direct=False:
      Do RAG first, then LLM.
    """
    def _rag_attempt(q: str) -> AnswerResult:
        r = retrieve(collection, q, k=TOP_K)
        # keep the raw top-k for transparency
        raw_docs = r["docs"]
        raw_sims = r["similarities"]
        # filter by threshold
        paired = [(doc, sim) for doc, sim in zip(raw_docs, raw_sims) if sim >= MIN_SIMILARITY]

        if not paired:
            return AnswerResult(text="I don't know.", source="IDK", retrieved=raw_docs, similarities=raw_sims)

        context_docs = [d for d, _ in paired]
        context_sims = [s for _, s in paired]
        rag = llm_with_context(q, context_docs).strip()
        if rag.lower() == "i don't know.":
            return AnswerResult(text=rag, source="IDK", retrieved=context_docs, similarities=context_sims)
        return AnswerResult(text=rag, source="RAG", retrieved=context_docs, similarities=context_sims)

    if prefer_direct:
        direct = llm_answer(f"Question: {question}\nProvide an accurate answer if you are reasonably sure.")
        if direct.lower() != "i don't know.":
            return AnswerResult(text=direct, source="LLM")
        return _rag_attempt(question)
    else:
        rag_res = _rag_attempt(question)
        if rag_res.source == "RAG":
            return rag_res
        direct = llm_answer(f"Question: {question}\nProvide an accurate answer if you are reasonably sure.")
        if direct.lower() != "i don't know.":
            return AnswerResult(text=direct, source="LLM")
        return AnswerResult(text="I don't know.", source="IDK")

# ---- CLI ----
if __name__ == "__main__":
    # Example docs to seed the Chroma collection
    docs = [
        "Azure AI pricing updated on Aug 2025: $0.001 per token",
        "AWS AI pricing July 2025: $0.002 per token",
        "Google AI pricing May 2025: $0.0015 per token"
    ]
    metadatas = [{"vendor": "Azure"}, {"vendor": "AWS"}, {"vendor": "Google"}]

    # Clean & ingest fresh
    try:
        client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_chroma_collection()
    ingest_documents(collection, docs, metadatas)

    print("🔹 Local RAG Assistant (Ollama + Chroma)")
    print("Type your question, or ':quit' to exit.")
    print("Commands: ':direct on' → LLM first, ':direct off' → RAG first.")
    print("          ':threshold <0..1>' → set min cosine similarity (current: {:.2f})\n".format(MIN_SIMILARITY))

    prefer_direct = True
    show_topk = True

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        low = q.lower()
        if low in [":quit", ":exit"]:
            print("👋 Goodbye!")
            break
        if low.startswith(":direct"):
            if "on" in low:
                prefer_direct = True
                print("⚙️ Direct-first mode enabled (LLM first, fallback to RAG).")
            elif "off" in low:
                prefer_direct = False
                print("⚙️ RAG-first mode enabled (retrieval first).")
            continue
        if low.startswith(":threshold"):
            try:
                _, val = q.split()
                valf = float(val)
                if 0.0 <= valf <= 1.0:
                    MIN_SIMILARITY = valf  # type: ignore
                    print(f"⚙️ MIN_SIMILARITY set to {MIN_SIMILARITY:.2f}")
                else:
                    print("Please provide a value between 0 and 1.")
            except Exception:
                print("Usage: :threshold 0.35")
            continue

        res = answer_question(q, collection, prefer_direct=prefer_direct)
        print(f"Assistant ({res.source}): {res.text}")

        if show_topk and res.source in ("RAG", "IDK") and res.retrieved:
            print("— Retrieved context (filtered by threshold):")
            sims = res.similarities or []
            for i, doc in enumerate(res.retrieved):
                sim = sims[i] if i < len(sims) else None
                if sim is not None:
                    print(f"  {i+1}. sim={sim:.3f}  {doc}")
                else:
                    print(f"  {i+1}. {doc}")
        print()
