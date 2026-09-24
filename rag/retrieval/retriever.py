from __future__ import annotations

import re
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

from core import config
from rag.chunking.chunker import chunk_text
from rag.embeddings.embedder import get_embedder
from rag.ingestion.loader import load_documents


@lru_cache(maxsize=1)
def policy_chunks() -> tuple:
    chunks = []
    for d in load_documents(config.POLICY_DIR):
        chunks += chunk_text(d["text"], d["source"])
    return tuple((c["source"], c["text"]) for c in chunks)


class Retriever:
    """Vector search (cosine) + lexical rerank. Build per request so live governance evidence
    for a model can be searched alongside the static policy documents."""

    def __init__(self, extra_chunks: list[dict] | None = None):
        base = [{"source": s, "text": t} for s, t in policy_chunks()]
        self.chunks = base + (extra_chunks or [])
        self.embedder = get_embedder()
        self.matrix = self.embedder.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 4, fetch: int = 12) -> list[dict]:
        sims = cosine_similarity(self.embedder.encode_query(query), self.matrix)[0]
        top = sims.argsort()[::-1][:fetch]
        q_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        ranked = []
        for i in top:
            terms = set(re.findall(r"[a-z0-9]{3,}", self.chunks[i]["text"].lower()))
            coverage = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            ranked.append({**self.chunks[i], "score": round(float(0.7 * sims[i] + 0.3 * coverage), 4)})
        ranked.sort(key=lambda d: d["score"], reverse=True)
        return [r for r in ranked[:k] if r["score"] > 0.02]
