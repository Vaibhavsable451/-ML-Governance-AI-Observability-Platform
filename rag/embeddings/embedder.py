"""Embedder abstraction. TF-IDF works offline; set EMBEDDER=sbert for sentence-transformers."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core import config


class TfidfEmbedder:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts)

    def encode_query(self, q: str):
        return self.vec.transform([q])


class SbertEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit_transform(self, texts):
        return np.asarray(self.m.encode(texts, normalize_embeddings=True))

    def encode_query(self, q: str):
        return np.asarray(self.m.encode([q], normalize_embeddings=True))


def get_embedder():
    return SbertEmbedder() if config.EMBEDDER == "sbert" else TfidfEmbedder()
