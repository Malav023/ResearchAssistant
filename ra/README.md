# ra_core — Phase 5 (hybrid RAG) + Phase 6 (Redis) 

## Layout
```
ra_core/rag/    types, chunking, bm25 (own), embedder, fusion (RRF), paper_ref, index (HybridIndex)
ra_core/cache/  keys, redis_cache (fail-open + breaker + single-flight), session_store
tests/          25 tests, no torch / no network / no real Redis needed
scripts/eval_rag.py   acceptance harness for real papers
```

## Colab (CPU is fine)
```python
# 1. DON'T reinstall torch/numpy/transformers. Only add small packages:
!pip -q install redis fakeredis pytest pytest-asyncio
# (sentence-transformers is usually preinstalled; only needed for the real embedder)
!unzip -q ra_core_phase5_6.zip -d /content/ra && cd /content/ra && python -m pytest -q
```
If `pip` ever prints "restart runtime" for numpy, stop and check `pip list | grep -E "numpy|torch|transformers"`
before/after — that is the classic Colab conflict.

## FastAPI wiring (Phase 8 preview)
```python
@asynccontextmanager
async def lifespan(app):
    s = load_settings()
    app.state.embedder = SentenceTransformerEmbedder(s.embed_model)   # load ONCE
    app.state.cache = RedisCache(s.redis_url, timeout_s=s.redis_timeout_s)
    app.state.sessions = SessionStore(app.state.cache, s)
    yield
    await app.state.cache.aclose()

# extraction (cache-aside + single-flight)
key = extraction_key(query, intent, s.extractor_version)
out, hit = await cache.get_or_compute(key, s.ttl_extract_s, lambda: run_extractor(query, intent))

# chat
idx = await sessions.get_index(sid, embedder)          # None -> "fetch papers first"
hits = await run_in_threadpool(idx.search, raw_query, k=5, extra_query=out["topic"])
context = render_context(hits)
```
