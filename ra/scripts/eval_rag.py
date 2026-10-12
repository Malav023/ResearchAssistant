"""Acceptance test for 'does the RAG meet our requirements'.

papers.json : [{"id","title","abstract","introduction"}, ...]  (3-5 REAL fetched papers)
qs.json     : [{"q": "...", "paper": 2, "must_contain": "optional substring"}, ...]  (~15 questions)

    python scripts/eval_rag.py --papers papers.json --questions qs.json --k 5
    python scripts/eval_rag.py ... --embedder hashing     # offline sanity run
Pass bar suggestion: hit@k >= 0.85 and paper-hit@k >= 0.95.
"""
import argparse, json, sys, time
sys.path.insert(0, ".")
from ra_core.rag import HashingEmbedder, HybridIndex, Paper

ap = argparse.ArgumentParser()
ap.add_argument("--papers", required=True); ap.add_argument("--questions", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
a = ap.parse_args()

if a.embedder == "hashing":
    emb = HashingEmbedder()
else:
    from ra_core.rag import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder(a.embedder)

papers = [Paper.from_dict(d) for d in json.load(open(a.papers))]
t = time.perf_counter(); idx = HybridIndex.build(papers, emb)
print(f"built {len(idx.chunks)} chunks in {(time.perf_counter()-t)*1000:.0f} ms")

qs = json.load(open(a.questions)); paper_hits = chunk_hits = 0; lat = []
for item in qs:
    t = time.perf_counter(); hits = idx.search(item["q"], k=a.k); lat.append((time.perf_counter()-t)*1000)
    ok_paper = any(h.chunk.paper_idx == item["paper"] for h in hits)
    needle = item.get("must_contain")
    ok_chunk = ok_paper and (not needle or any(needle.lower() in h.chunk.text.lower() for h in hits))
    paper_hits += ok_paper; chunk_hits += ok_chunk
    if not ok_chunk:
        print("MISS:", item["q"], "->", [(h.chunk.paper_idx, h.chunk.section) for h in hits])
n = len(qs); lat.sort()
print(f"paper-hit@{a.k}: {paper_hits/n:.2f}   hit@{a.k}: {chunk_hits/n:.2f}   "
      f"search p50={lat[n//2]:.0f} ms max={lat[-1]:.0f} ms")
