"""Minimal BM25 (Okapi, Lucene-style IDF). ~40 lines, only depends on numpy.

Why not rank_bm25: its classic IDF goes negative/zero for terms that appear in
more than half the documents. With ~20 chunks that happens constantly, which
silently drops useful terms. Lucene's log(1 + ...) form is always positive.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be been but by can did do does for from had has have how i in into is it its "
    "me my of on or our paper papers please say says so some than that the their them then there these "
    "they this to us was we were what when where which who why will with would you your about tell".split()
)


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self._tf = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for c in self._tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            tf = np.array([c.get(t, 0) for c in self._tf], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
