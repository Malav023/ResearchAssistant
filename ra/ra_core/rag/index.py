from __future__ import annotations

import json
import struct

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import chunk_paper
from .embedder import Embedder
from .fusion import rrf
from .paper_ref import extract_paper_refs
from .types import Chunk, Hit, Paper, papers_fingerprint

_MAGIC = b"HIX1"


class IndexStale(Exception):
    """Serialized index doesn't match the current embedder or paper set -> rebuild."""


def _rank(scores: np.ndarray, allowed: np.ndarray, positive_only: bool = False) -> list[int]:
    idx = np.flatnonzero(allowed)
    if positive_only:
        idx = idx[scores[idx] > 0]  # zero BM25 score = no lexical evidence; don't let it vote
    return idx[np.argsort(-scores[idx], kind="stable")].tolist()


class HybridIndex:
    """In-memory BM25 + dense retrieval fused with RRF over ~20 chunks. No vector DB."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder,
                 fingerprint: str, n_papers: int, rrf_k: int = 60):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb = (embeddings / np.where(norms == 0, 1, norms)).astype(np.float32)
        self.chunks, self.embedder = chunks, embedder
        self.fingerprint, self.n_papers, self.rrf_k = fingerprint, n_papers, rrf_k
        self.bm25 = BM25([tokenize(c.index_text) for c in chunks])  # rebuilt, never cached: ~free

    # ---------- build ----------
    @classmethod
    def build(cls, papers: list[Paper], embedder: Embedder, *, fingerprint: str | None = None,
              rrf_k: int = 60, **chunk_kwargs) -> "HybridIndex":
        chunks: list[Chunk] = []
        for i, p in enumerate(papers, 1):
            chunks += chunk_paper(p, i, **chunk_kwargs)
        if not chunks:
            raise ValueError("No indexable text (papers have empty abstracts/introductions).")
        emb = embedder.encode_passages([c.index_text for c in chunks])
        return cls(chunks, emb, embedder, fingerprint or papers_fingerprint(papers), len(papers), rrf_k)

    # ---------- search ----------
    def search(self, query: str, *, k: int = 5, extra_query: str | None = None,
               paper_idxs: set[int] | None = None, auto_paper_refs: bool = True) -> list[Hit]:
        """query: the student's RAW text (keeps 'the second paper'). extra_query: extracted topic,
        added as a half-weight signal. Blocking (embeds the query) -> call via a threadpool in FastAPI."""
        refs = set(paper_idxs) if paper_idxs else set()
        if not refs and auto_paper_refs:
            refs = extract_paper_refs(query, self.n_papers)
        allowed = np.ones(len(self.chunks), dtype=bool)
        if refs:
            allowed = np.array([c.paper_idx in refs for c in self.chunks])
            if not allowed.any():
                allowed, refs = np.ones(len(self.chunks), dtype=bool), set()

        queries = [(query, 1.0)]
        if extra_query and extra_query.strip().lower() != query.strip().lower():
            queries.append((extra_query, 0.5))

        rankings: list[list[int]] = []
        weights: list[float] = []
        for q, w in queries:
            dense = self.emb @ self.embedder.encode_query(q)
            sparse = self.bm25.scores(tokenize(q))
            rankings += [_rank(dense, allowed), _rank(sparse, allowed, positive_only=True)]
            weights += [w, w]

        fused = rrf(rankings, k=self.rrf_k, weights=weights)
        chosen = fused[:k]

        # Coverage guarantee: "compare paper 1 and 3" must show both papers.
        if len(refs) > 1:
            for p in sorted(refs):
                if any(self.chunks[c].paper_idx == p for c, _ in chosen):
                    continue
                best = next((cs for cs in fused if self.chunks[cs[0]].paper_idx == p), None)
                if best is None:
                    continue
                for j in range(len(chosen) - 1, -1, -1):
                    pj = self.chunks[chosen[j][0]].paper_idx
                    if sum(self.chunks[c].paper_idx == pj for c, _ in chosen) > 1:
                        chosen[j] = best
                        break
                else:
                    chosen.append(best)
            chosen.sort(key=lambda cs: (-cs[1], cs[0]))

        d_pos = {d: r for r, d in enumerate(rankings[0])}
        s_pos = {d: r for r, d in enumerate(rankings[1])}
        return [Hit(self.chunks[c], s, d_pos.get(c), s_pos.get(c)) for c, s in chosen]

    # ---------- (de)serialization for Redis ----------
    def to_bytes(self) -> bytes:
        header = json.dumps({
            "embed_model": self.embedder.name, "dim": int(self.emb.shape[1]), "n": len(self.chunks),
            "n_papers": self.n_papers, "fingerprint": self.fingerprint, "rrf_k": self.rrf_k,
            "chunks": [c.to_dict() for c in self.chunks],
        }, ensure_ascii=False).encode()
        return _MAGIC + struct.pack("<I", len(header)) + header + self.emb.astype("<f4").tobytes()

    @classmethod
    def from_bytes(cls, blob: bytes, embedder: Embedder, *, expected_fingerprint: str | None = None
                   ) -> "HybridIndex":
        try:
            if blob[:4] != _MAGIC:
                raise IndexStale("bad magic")
            (hlen,) = struct.unpack("<I", blob[4:8])
            h = json.loads(blob[8:8 + hlen])
            if h["embed_model"] != embedder.name:
                raise IndexStale(f"embedder changed: {h['embed_model']} != {embedder.name}")
            if expected_fingerprint and h["fingerprint"] != expected_fingerprint:
                raise IndexStale("paper set changed")
            emb = np.frombuffer(blob, dtype="<f4", offset=8 + hlen).reshape(h["n"], h["dim"])
            chunks = [Chunk.from_dict(c) for c in h["chunks"]]
        except IndexStale:
            raise
        except Exception as e:  # truncated / corrupt blob
            raise IndexStale(f"corrupt index: {e}") from e
        return cls(chunks, emb, embedder, h["fingerprint"], h["n_papers"], h.get("rrf_k", 60))


def render_context(hits: list[Hit]) -> str:
    """Prompt-ready context. Paper number + title next to every chunk so the LLM can
    resolve 'the second paper' and cite as [Paper N]."""
    ordered = sorted(hits, key=lambda h: (h.chunk.paper_idx, h.chunk.chunk_id))
    return "\n\n".join(
        f'[Paper {h.chunk.paper_idx} | "{h.chunk.title}" | {h.chunk.section}]\n{h.chunk.text}'
        for h in ordered
    )
