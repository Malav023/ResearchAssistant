"""Real main.app: lifespan, indexes, cookie auth (bcrypt + JWT), assistant routes. Only Mongo (in-memory),
OpenAlex and the LLM are faked. No network, no Redis, no models."""
import httpx
import pytest
from mongomock_motor import AsyncMongoMockClient

import db as dbmod
from conftest import FakeLLM, make_papers


@pytest.fixture
async def live(monkeypatch):
    monkeypatch.setattr(dbmod, "_client", AsyncMongoMockClient())
    monkeypatch.setenv("ML_BACKEND", "stub")
    import main
    async with main.app.router.lifespan_context(main.app):
        svc = main.app.state.svc

        async def fetch(topic, limit):
            return make_papers(limit)
        svc.fetch_papers = fetch
        svc.llm = FakeLLM()
        svc.cfg = __import__("dataclasses").replace(svc.cfg, llm_api_key="k", llm_chat_model="m", debug=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
            yield c, svc


async def test_full_flow_with_real_auth(live):
    c, svc = live
    assert (await c.post("/api/assistant/message", json={"message": "hi"})).status_code == 401
    r = await c.post("/api/auth/register", json={"name": "Ada", "email": "ada@example.com", "password": "correct horse"})
    assert r.status_code == 200 and "ra_session" in c.cookies

    d = (await c.post("/api/assistant/message", json={"message": "find the latest papers on federated learning privacy"})).json()
    assert d["action"] == "papers" and d["topic"] == "federated learning privacy" and len(d["papers"]) == 5
    sid = d["session_id"]

    d2 = (await c.post("/api/assistant/message", json={"session_id": sid, "message": "what is the evaluation setup in paper 2?"})).json()
    assert d2["action"] == "answer" and d2["message"]["citations"][0]["paper"] == 2
    assert d2["debug"]["retrieval"]["refs_detected"] == [2]

    sessions = (await c.get("/api/sessions")).json()["sessions"]          # existing sidebar route sees the new session
    assert [s["id"] for s in sessions] == [sid] and sessions[0]["query"].startswith("find the latest")
    detail = (await c.get(f"/api/assistant/sessions/{sid}")).json()
    assert len(detail["messages"]) == 4 and len(detail["papers"]) == 5

    deep = (await c.get("/api/health/deep")).json()
    assert deep["ml"]["backend"] == "stub" and deep["redis"] == "disabled" and deep["embedder"]["backend"] == "hashing"
