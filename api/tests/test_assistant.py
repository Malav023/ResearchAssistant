import dataclasses

import httpx

from assistant.ratelimit import RateLimiter
from llm.client import LLMError
from ml.runtime import MLError
from ra_core.cache import SessionStore
from conftest import make_papers


async def _fetch(env, text="find papers on federated learning privacy"):
    env.ml.label = "fetch_sources"
    r = await env.post(text)
    assert r.status_code == 200, r.text
    return r.json()


async def test_unclear_asks_clarifying_question(env):
    env.ml.label = "unclear"
    d = (await env.post("transformers")).json()
    assert d["action"] == "clarify" and len(d["suggestions"]) == 3
    assert env.ml.calls["extract"] == 0 and not env.llm.calls


async def test_low_confidence_is_gated_and_logged(env):
    env.ml.label, env.ml.conf = "chat", 0.4
    d = (await env.post("hmm what about it")).json()
    assert d["action"] == "clarify" and d["intent"] == "unclear"
    c = d["debug"]["classifier"]
    assert c["gated"] is True and c["top"] == "chat" and abs(sum(c["probs"].values()) - 1) < 1e-6


async def test_fetch_sources_flow_and_caches(env):
    d = await _fetch(env)
    assert d["action"] == "papers" and d["topic"] == "federated learning privacy"
    assert [p["n"] for p in d["papers"]] == [1, 2, 3, 4, 5] and d["papers"][0]["abstract"]
    assert not env.llm.calls                                   # no LLM on the fetch path
    assert d["debug"]["extractor"]["cache_hit"] is False and d["debug"]["sources"]["n"] == 5
    await _fetch(env)                                          # identical text, new session
    assert env.ml.calls["extract"] == 1 and len(env.fetch_calls) == 1
    d3 = await _fetch(env, "pls pull FL privacy papers")       # new text, same extracted topic
    assert env.ml.calls["extract"] == 2 and len(env.fetch_calls) == 1
    assert d3["debug"]["sources"]["cache_hit"] is True


async def test_empty_results_not_cached(env):
    env.papers = []
    d = await _fetch(env)
    assert d["papers"] == [] and "couldn't find" in d["message"]["content"]
    await _fetch(env)
    assert len(env.fetch_calls) == 2


async def test_extractor_failure_falls_back(env):
    env.ml.fail_extract = RuntimeError("boom")
    d = await _fetch(env, "find papers on quantum computing")
    assert "extractor" in d["degraded"] and d["topic"] == "quantum computing" and d["papers"]


async def test_extractor_timeout_falls_back(env):
    env.svc.cfg = dataclasses.replace(env.svc.cfg, extractor_timeout_s=0.05)
    env.ml.extract_delay = 0.5
    d = await _fetch(env, "find papers on graph neural networks")
    assert "extractor" in d["degraded"] and d["debug"]["extractor"]["fallback"] is True


async def test_classifier_failure_uses_keyword_fallback(env):
    env.ml.fail_classify = MLError("ML_UNREACHABLE", "colab is down")
    d = await _fetch(env, "find recent papers on graph neural networks")
    assert "classifier" in d["degraded"] and d["action"] == "papers"
    assert d["debug"]["classifier"]["synthetic"] is True


async def test_chat_without_papers(env):
    env.ml.label = "chat"
    d = (await env.post("what does paper 2 say")).json()
    assert d["action"] == "needs_papers" and not env.llm.calls


async def test_chat_is_grounded_and_citations_validated(env):
    sid = (await _fetch(env))["session_id"]
    env.ml.label = "chat"
    extract_before = env.ml.calls["extract"]
    env.llm.reply = "It uses identical budgets [Paper 2] and more [Paper 9]."
    d = (await env.post("what is the evaluation setup in the second paper", sid)).json()
    assert d["action"] == "answer" and "[Paper 2]" in d["message"]["content"] and "[Paper 9]" not in d["message"]["content"]
    assert [c["paper"] for c in d["message"]["citations"]] == [2] and "citations" in d["degraded"]
    user_msg = env.llm.calls[-1]["messages"][-1]["content"]
    assert "[Paper 2 |" in user_msg and "QUESTION:" in user_msg and env.llm.calls[-1]["model"] == "chat-m"
    assert env.ml.calls["extract"] == extract_before                 # extractor skipped for chat
    assert d["debug"]["retrieval"]["refs_detected"] == [2] and d["debug"]["extractor"]["ran"] is False
    # follow-up carries history
    await env.post("and what about their limitations?", sid)
    roles = [m["role"] for m in env.llm.calls[-1]["messages"]]
    assert roles[0] == "system" and "assistant" in roles and roles[-1] == "user"


async def test_chat_survives_redis_loss(env):
    sid = (await _fetch(env))["session_id"]
    await env.svc.cache._client.flushall()
    env.svc.sessions = SessionStore(env.svc.cache, env.cfg.ra)        # simulates a server restart
    env.ml.label = "chat"
    d = (await env.post("what does paper 1 claim", sid)).json()
    assert d["action"] == "answer"


async def test_chat_llm_failure_is_typed(env):
    sid = (await _fetch(env))["session_id"]
    env.ml.label = "chat"
    env.llm.error = LLMError("LLM_RATE_LIMITED", "busy", status=429, retryable=True)
    r = await env.post("what does paper 1 claim", sid)
    e = r.json()["error"]
    assert r.status_code == 429 and e["code"] == "LLM_RATE_LIMITED" and e["stage"] == "llm" and e["retryable"] is True
    assert e["trace_id"] == r.headers["x-trace-id"]


async def test_embedder_failure_falls_back_to_stuffing(env):
    class Broken:
        name = "broken"
        def encode_passages(self, t): raise MLError("EMBED_UNREACHABLE", "down")
        def encode_query(self, t): raise MLError("EMBED_UNREACHABLE", "down")
    sid = (await _fetch(env))["session_id"]
    env.svc.embedder = Broken()
    env.ml.label = "chat"
    d = (await env.post("what does paper 3 say", sid)).json()
    assert d["action"] == "answer" and "embedder" in d["degraded"] and d["debug"]["retrieval"]["mode"] == "stuff"


async def test_papers_without_abstracts(env):
    env.papers = make_papers(abstracts=False)
    sid = (await _fetch(env))["session_id"]
    env.ml.label = "chat"
    d = (await env.post("what does paper 1 say", sid)).json()
    assert "abstracts" in d["message"]["content"] and not env.llm.calls


async def test_report_flow_appends_server_built_references(env):
    env.ml.label = "create_report"
    env.llm.reply = "## Overview\nIntro [Paper 1].\n\n## Paper by paper\n### Paper 3: X\nText [Paper 3, Paper 7]."
    d = (await env.post("write a report on federated learning privacy")).json()
    assert d["action"] == "report" and len(d["papers"]) == 5
    body = d["message"]["content"]
    assert "## References" in body and "**[Paper 5]**" in body and "Paper 7" not in body
    assert env.llm.calls[-1]["model"] == "rep-m"


async def test_report_degrades_to_sources_when_llm_down(env):
    env.ml.label = "create_report"
    env.llm.error = LLMError("LLM_UNAVAILABLE", "provider down.", status=502)
    d = (await env.post("write a report on federated learning privacy")).json()
    assert d["action"] == "papers" and "llm" in d["degraded"] and len(d["papers"]) == 5
    assert "couldn't draft" in d["message"]["content"]


async def test_sources_upstream_error_cleans_up_empty_session(env):
    env.fetch_error = httpx.ConnectError("no network")
    r = await env.post("find papers on anything")
    assert r.status_code == 502 and r.json()["error"]["code"] == "SOURCES_UPSTREAM"
    assert await env.db["search_sessions"].count_documents({}) == 0


async def test_session_ownership(env):
    sid = (await _fetch(env))["session_id"]
    from bson import ObjectId
    env.user = {"_id": ObjectId(), "name": "Other", "email": "o@example.com"}
    r = await env.post("hello", sid)
    assert r.status_code == 404 and r.json()["error"]["code"] == "SESSION_NOT_FOUND"
    assert (await env.client.get(f"/api/assistant/sessions/{sid}")).status_code == 404


async def test_session_restore(env):
    sid = (await _fetch(env))["session_id"]
    env.ml.label = "chat"
    await env.post("what does paper 1 claim", sid)
    d = (await env.client.get(f"/api/assistant/sessions/{sid}")).json()
    assert len(d["papers"]) == 5 and [m["role"] for m in d["messages"]] == ["user", "assistant", "user", "assistant"]
    assert d["messages"][3]["action"] == "answer"


async def test_rate_limit_and_validation(env):
    env.svc.limiter = RateLimiter(2)
    for _ in range(2):
        assert (await env.post("find papers on x")).status_code == 200
    r = await env.post("find papers on x")
    assert r.status_code == 429 and r.json()["error"]["code"] == "RATE_LIMITED"
    env.svc.limiter = RateLimiter(1000)
    for bad in ("", "x" * 2001):
        r = await env.post(bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION"


async def test_traces_debug_endpoints_and_feedback(env):
    d = await _fetch(env)
    t = (await env.client.get("/api/debug/traces")).json()["traces"]
    assert t and t[0]["trace_id"] == d["trace_id"] and t[0]["classifier"]["p_top"] == 0.93
    assert (await env.client.get("/api/debug/traces?max_conf=0.5")).json()["traces"] == []
    assert (await env.client.post("/api/assistant/feedback", json={"trace_id": d["trace_id"], "correct_intent": "chat"})).status_code == 200
    one = (await env.client.get(f"/api/debug/traces/{d['trace_id']}")).json()
    assert one["feedback"]["correct_intent"] == "chat"
    env.svc.cfg = dataclasses.replace(env.svc.cfg, debug=False)
    assert (await env.client.get("/api/debug/traces")).status_code == 404
    assert "debug" not in (await _fetch(env)) or (await _fetch(env))["debug"] is None


async def test_health_deep(env):
    d = (await env.client.get("/api/health/deep")).json()
    assert d["ml"]["backend"] == "fake" and d["llm"]["configured"] is True and d["redis"] is True
