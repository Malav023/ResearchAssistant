"""HTTP client for the Colab model server (ml_service.py)."""
from __future__ import annotations

import httpx
import numpy as np

from .runtime import ClassifyResult, ExtractResult, MLError


def _headers(token: str) -> dict:
    h = {"ngrok-skip-browser-warning": "true"}   # harmless elsewhere; avoids the ngrok interstitial
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise MLError("ML_AUTH", "model server rejected the token (check ML_SERVICE_TOKEN)", retryable=False)
    if resp.status_code >= 500:
        raise MLError("ML_UPSTREAM", f"model server error {resp.status_code}")
    if resp.status_code >= 400:
        raise MLError("ML_BAD_REQUEST", f"model server rejected request ({resp.status_code})", retryable=False)


class RemoteMLRuntime:
    name = "remote"

    def __init__(self, base_url: str, token: str, timeout_s: float = 25.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not base_url:
            raise ValueError("ML_REMOTE_URL is empty (start the Colab server and paste its URL into .env)")
        self._c = httpx.AsyncClient(base_url=base_url, headers=_headers(token), transport=transport,
                                    timeout=httpx.Timeout(timeout_s, connect=8.0))

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            r = await self._c.post(path, json=payload)
        except httpx.TimeoutException as e:
            raise MLError("ML_TIMEOUT", f"model server timed out on {path}") from e
        except httpx.TransportError as e:
            raise MLError("ML_UNREACHABLE", f"cannot reach model server ({type(e).__name__}); is the Colab cell still running?") from e
        _raise_for(r)
        try:
            return r.json()
        except ValueError as e:
            raise MLError("ML_BAD_RESPONSE", "model server returned non-JSON") from e

    async def classify(self, text: str) -> ClassifyResult:
        d = await self._post("/classify", {"text": text})
        try:
            return ClassifyResult(d["label"], float(d["confidence"]), dict(d.get("probs") or {}),
                                  version=d.get("version", ""))
        except (KeyError, TypeError, ValueError) as e:
            raise MLError("ML_BAD_RESPONSE", f"malformed /classify response: {e}") from e

    async def extract(self, text: str, intent: str) -> ExtractResult:
        d = await self._post("/extract", {"text": text, "intent": intent})
        if not isinstance(d.get("topic"), str):
            raise MLError("ML_BAD_RESPONSE", "malformed /extract response")
        return ExtractResult(topic=d["topic"], avg_logprob=d.get("avg_logprob"), version=d.get("version", ""))

    async def health(self) -> dict:
        try:
            r = await self._c.get("/health", timeout=5.0)
            return {"backend": "remote", "reachable": True, **r.json()}
        except Exception as e:
            return {"backend": "remote", "reachable": False, "ok": False, "error": f"{type(e).__name__}: {e}"}

    async def warmup(self) -> None:
        h = await self.health()
        if not h.get("ok"):
            raise MLError("ML_NOT_READY", f"model server not ready: {h.get('error') or h}")

    async def aclose(self) -> None:
        await self._c.aclose()


class RemoteEmbedder:
    """Sync embedder (HybridIndex calls it from worker threads) backed by the model server's /embed."""

    def __init__(self, base_url: str, token: str, model_name: str, timeout_s: float = 15.0,
                 transport: httpx.BaseTransport | None = None, batch_size: int = 64):
        self.name = model_name
        self._batch = batch_size
        self._c = httpx.Client(base_url=base_url, headers=_headers(token), transport=transport,
                               timeout=httpx.Timeout(timeout_s, connect=8.0))

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            try:
                r = self._c.post("/embed", json={"texts": texts[i:i + self._batch], "kind": kind})
            except httpx.HTTPError as e:
                raise MLError("EMBED_UNREACHABLE", f"embedding server unreachable ({type(e).__name__})") from e
            if r.status_code >= 400:
                raise MLError("EMBED_FAILED", f"embedding server error {r.status_code}")
            d = r.json()
            if d.get("model") != self.name:
                raise MLError("EMBED_MODEL_MISMATCH",
                              f"server embeds with {d.get('model')!r}, API expects {self.name!r} (set EMBED_MODEL on both)",
                              retryable=False)
            out += d["vectors"]
        return np.asarray(out, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage")

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query")[0]
