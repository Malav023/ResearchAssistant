from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from .bm25 import tokenize


class Embedder(Protocol):
    name: str

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """bge-small / e5-small via sentence-transformers. Load ONCE at app startup."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu",
                 batch_size: int = 32, max_seq_length: int = 384):
        from sentence_transformers import SentenceTransformer  # lazy: keeps tests torch-free

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length
        low = model_name.lower()
        if "e5" in low:
            self._q_prefix, self._p_prefix = "query: ", "passage: "
        elif "bge" in low:
            self._q_prefix, self._p_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self._q_prefix = self._p_prefix = ""

    def _enc(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._enc([self._p_prefix + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._enc([self._q_prefix + text])[0]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. TEST DOUBLE / offline dev only."""

    def __init__(self, dim: int = 256):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)
