import numpy as np
import pytest
from ra_core.rag import HashingEmbedder, HybridIndex, IndexStale, Paper, render_context
from ra_core.rag.bm25 import BM25, tokenize
from ra_core.rag.chunking import chunk_paper, split_sentences
from ra_core.rag.fusion import rrf
from ra_core.rag.paper_ref import extract_paper_refs


def test_rrf_basic_and_weights():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0][0] == 1
    assert rrf([[1, 2], [2, 1]], weights=[1.0, 0.1])[0][0] == 1


def test_bm25_idf_never_negative_in_tiny_corpus():
    docs = [tokenize("attention attention model"), tokenize("attention model"), tokenize("graph model")]
    bm = BM25(docs)
    assert all(v > 0 for v in bm.idf.values())
    assert bm.scores(tokenize("graph"))[2] > bm.scores(tokenize("graph"))[0]


def test_sentence_split_keeps_abbreviations():
    s = split_sentences("Gradients leak (Zhu et al. 2019). We add noise, e.g. Gaussian. Done.")
    assert len(s) == 3 and s[0].endswith("2019).")


def test_chunking_limits(papers):
    for i, p in enumerate(papers, 1):
        ch = chunk_paper(p, i)
        assert 1 <= len(ch) <= 4 and all(c.paper_idx == i for c in ch)
    ch = chunk_paper(Paper("x", "T", "One sentence abstract only."), 1)
    assert len(ch) == 1 and ch[0].section == "abstract"
    long_intro = " ".join(f"Sentence number {i} has several words in it." for i in range(300))
    assert len(chunk_paper(Paper("y", "T", "Abs.", long_intro), 1)) <= 4


@pytest.mark.parametrize("q,n,exp", [
    ("what does the second paper say about evaluation", 5, {2}),
    ("compare the first and third papers", 5, {1, 3}),
    ("summarise paper 4", 5, {4}),
    ("paper two limitations", 5, {2}),
    ("the last paper", 5, {5}),
    ("first, explain how it works", 5, set()),
    ("what about the 9th paper", 5, set()),
])
def test_paper_refs(q, n, exp):
    assert extract_paper_refs(q, n) == exp


def test_search_targets_referenced_paper(papers):
    idx = HybridIndex.build(papers, HashingEmbedder())
    hits = idx.search("what is the evaluation setup in the second paper", k=4)
    assert hits and all(h.chunk.paper_idx == 2 for h in hits)


def test_search_topical_and_extra_query(papers):
    idx = HybridIndex.build(papers, HashingEmbedder())
    top = idx.search("why do models lose old skills", k=3, extra_query="catastrophic forgetting")[0]
    assert top.chunk.paper_idx == 3


def test_coverage_for_multi_paper_queries(papers):
    idx = HybridIndex.build(papers, HashingEmbedder())
    hits = idx.search("compare the first and third paper", k=2)
    assert {h.chunk.paper_idx for h in hits} == {1, 3}


def test_roundtrip_and_staleness(papers):
    emb = HashingEmbedder()
    idx = HybridIndex.build(papers, emb)
    idx2 = HybridIndex.from_bytes(idx.to_bytes(), emb, expected_fingerprint=idx.fingerprint)
    q = "differential privacy noise"
    assert [h.chunk.chunk_id for h in idx.search(q)] == [h.chunk.chunk_id for h in idx2.search(q)]
    with pytest.raises(IndexStale):
        HybridIndex.from_bytes(idx.to_bytes(), HashingEmbedder(dim=128))
    with pytest.raises(IndexStale):
        HybridIndex.from_bytes(idx.to_bytes(), emb, expected_fingerprint="other")
    with pytest.raises(IndexStale):
        HybridIndex.from_bytes(idx.to_bytes()[:50], emb)


def test_render_context(papers):
    idx = HybridIndex.build(papers, HashingEmbedder())
    ctx = render_context(idx.search("federated learning", k=3))
    assert ctx.startswith("[Paper ")
