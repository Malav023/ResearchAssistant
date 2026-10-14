import httpx
import pytest

import config


async def test_main_imports_and_exposes_routes():
    import main
    paths = set(main.app.openapi()["paths"])
    for p in ("/health", "/api/graph", "/api/papers", "/api/assistant/message",
              "/api/assistant/sessions/{sid}", "/api/health/deep", "/api/auth/login", "/api/sessions"):
        assert p in paths, p
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        r = await c.get("/health")
        assert r.json() == {"status": "ok"} and len(r.headers["x-trace-id"]) == 12
        r = await c.post("/api/assistant/message", json={"message": "hi"})     # no cookie
        assert r.status_code == 401


def test_remote_backend_without_url_fails_loudly():
    from dataclasses import replace
    from assistant.services import make_ml
    with pytest.raises(ValueError, match="ML_REMOTE_URL"):
        make_ml(replace(config.get_config(), ml_backend="remote", ml_remote_url=""))
    with pytest.raises(ValueError):
        make_ml(replace(config.get_config(), ml_backend="nonsense"))
