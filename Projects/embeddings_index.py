import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from google import genai  # pip install google-genai

# ---- Data structures ----

@dataclass
class ChunkEmbedding:
    id: str
    page_range: Tuple[int, int]
    text: str
    vector: List[float]

# ---- Helpers ----

def cosine_similarity(a: List[float], b: List[float]) -> float:
    # TODO: implement cosine similarity safely
    # hint: dot(a, b) / (||a|| * ||b||)
    if len(a) != len(b):
        raise ValueError("Embedding size mismatch")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

def get_gemini_client() -> "genai.Client":
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment")
    # For google-genai, just constructing Client() picks up env key
    client = genai.Client()
    return client

def embed_chunks(
    chunks: List[Dict],
    model: str = "gemini-embedding-001",  # or "gemini/text-embedding-004"
) -> List[ChunkEmbedding]:
    """
    Take your chunk dicts and return ChunkEmbedding objects with vectors.
    """
    client = get_gemini_client()
    texts = [c["text"] for c in chunks]

    # TODO: call Gemini embeddings API in batch for these texts
    # use client.models.embed_content(...)
    # hint: contents=texts

    # Pseudo-structure of result from docs:
    # result.embeddings is a list; each item has .values (the vector)
    # https://ai.google.dev/gemini-api/docs/embeddings
    result = client.models.embed_content(
        model=model,
        contents=texts,
    )

    embeddings: List[ChunkEmbedding] = []
    for chunk, emb in zip(chunks, result.embeddings):
        vector = emb.values  # or emb["values"] depending on SDK version
        embeddings.append(
            ChunkEmbedding(
                id=chunk["id"],
                page_range=tuple(chunk["page_range"]),
                text=chunk["text"],
                vector=vector,
            )
        )
    return embeddings

def embed_query(query: str, model: str = "gemini-embedding-001") -> List[float]:
    client = get_gemini_client()
    result = client.models.embed_content(
        model=model,
        contents=[query],
    )
    # Take the first embedding
    return result.embeddings[0].values

def search_chunks(
    query: str,
    chunk_embeddings: List[ChunkEmbedding],
    top_k: int = 5,
    model: str = "gemini-embedding-001",
) -> List[Tuple[ChunkEmbedding, float]]:
    """
    Embed the query, compute cosine similarity against all chunks,
    return top_k (chunk, score) pairs sorted by score desc.
    """
    q_vec = embed_query(query, model=model)
    scored: List[Tuple[ChunkEmbedding, float]] = []
    for ce in chunk_embeddings:
        score = cosine_similarity(q_vec, ce.vector)
        scored.append((ce, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    from pathlib import Path
    from pdf_rag import load_pdf_text, chunk_pages  # adjust import to your file name

    pdf_path = Path("policy.pdf")
    pages = load_pdf_text(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Loaded {len(pages)} pages, {len(chunks)} chunks")

    chunk_embs = embed_chunks(chunks)
    print(f"Embedded {len(chunk_embs)} chunks")

    results = search_chunks(
        "What is the cancellation policy?",
        chunk_embs,
        top_k=3,
    )
    for ce, score in results:
        print("-----")
        print(ce.id, ce.page_range, f"score={score:.3f}")
        print(ce.text[:300].replace("\n", " "))
