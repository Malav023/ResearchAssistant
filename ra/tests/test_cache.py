import asyncio
import fakeredis.aioredis
import redis.asyncio as aioredis
from ra_core.cache import RedisCache, SessionStore, extraction_key, normalize_text, sources_key
from ra_core.rag import HashingEmbedder
from ra_core.settings import Settings


def fake_cache(**kw):
    return RedisCache(client=fakeredis.aioredis.FakeRedis(), **kw)


def test_keys():
    assert normalize_text("  Federated   Learning!! ") == "federated learning"
    assert normalize_text("C++") != normalize_text("C#") != normalize_text("C")
    a = extraction_key("Hello, World", "chat", "v1")
    assert a == extraction_key("hello world", "chat", "v1")
    assert a != extraction_key("hello world", "fetch_sources", "v1")
    assert a != extraction_key("hello world", "chat", "v2")
    assert sources_key("LLM agents", limit=5) != sources_key("LLM agents", limit=10)


async def test_json_roundtrip_and_ttl():
    c = fake_cache()
    assert await c.get_json("k") is None
    assert await c.set_json("k", {"topic": "t", "brief": "b"}, 60)
    assert await c.get_json("k") == {"topic": "t", "brief": "b"}
    assert 0 < await c._client.ttl("k") <= 60


async def test_disabled_cache_is_all_misses():
    c = RedisCache("")
    assert not c.enabled and await c.get_json("x") is None and not await c.set_json("x", 1, 5)


async def test_fail_open_and_breaker():
    c = RedisCache(client=aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.05),
                   timeout_s=0.05, breaker_threshold=2, breaker_cooldown_s=60)
    for _ in range(4):
        assert await c.get_json("k") is None      # never raises
    assert c.stats["error"] == 2 and c.stats["skipped"] >= 2  # breaker opened after 2 failures
    val, cached = await c.get_or_compute("k", 5, lambda: _const({"a": 1}))
    assert val == {"a": 1} and cached is False    # still serves via compute


async def _const(v):
    return v


async def test_single_flight():
    c = fake_cache()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"v": 1}

    res = await asyncio.gather(*[c.get_or_compute("hot", 60, compute) for _ in range(10)])
    assert calls == 1 and all(r[0] == {"v": 1} for r in res)
    assert (await c.get_or_compute("hot", 60, compute))[1] is True and calls == 1


async def test_single_flight_error_propagates_and_none_not_cached():
    c = fake_cache()

    async def boom():
        raise RuntimeError("x")

    for _ in range(2):
        try:
            await c.get_or_compute("e", 5, boom)
            assert False
        except RuntimeError:
            pass
    assert (await c.get_or_compute("n", 5, lambda: _const(None)))[0] is None
    assert await c.get_json("n") is None


async def test_session_store_flow(papers):
    s = Settings(history_max_msgs=3)
    store, emb = SessionStore(fake_cache(), s), HashingEmbedder()
    assert await store.get_papers("s1") is None and await store.get_index("s1", emb) is None
    await store.save_papers("s1", papers)
    idx = await store.get_index("s1", emb)
    assert idx and await store.get_index("s1", emb) is idx          # L1 reuse

    # fresh process: no local state, rebuilt from Redis blob without re-embedding
    store2 = SessionStore(store.cache, s)
    calls = {"n": 0}
    orig = emb.encode_passages
    emb.encode_passages = lambda t: (calls.__setitem__("n", calls["n"] + 1), orig(t))[1]
    idx2 = await store2.get_index("s1", emb)
    assert idx2 and calls["n"] == 0 and idx2.fingerprint == idx.fingerprint

    # re-fetch replaces papers and invalidates index
    await store.save_papers("s1", papers[:2])
    assert (await store.get_index("s1", emb)).n_papers == 2

    for i in range(5):
        await store.append_turn("s1", "user", f"m{i}")
    assert [m["content"] for m in await store.get_history("s1")] == ["m2", "m3", "m4"]


async def test_session_survives_redis_outage(papers):
    dead = RedisCache(client=aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.05),
                      timeout_s=0.05)
    store, emb = SessionStore(dead, Settings()), HashingEmbedder()
    await store.save_papers("s", papers)
    assert await store.get_index("s", emb) is not None
    await store.append_turn("s", "user", "hi")
    assert (await store.get_history("s"))[0]["content"] == "hi"


async def test_concurrent_index_build_runs_once(papers):
    store, emb = SessionStore(fake_cache(), Settings()), HashingEmbedder()
    await store.save_papers("s", papers)
    n = {"c": 0}
    orig = emb.encode_passages
    emb.encode_passages = lambda t: (n.__setitem__("c", n["c"] + 1), orig(t))[1]
    idxs = await asyncio.gather(*[store.get_index("s", emb) for _ in range(8)])
    assert n["c"] == 1 and len({id(i) for i in idxs}) == 1
