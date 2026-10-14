"""OpenAI-compatible chat client (Groq, HF router, OpenRouter, ...). httpx only, no SDK.

Retry policy: transient failures (429 / 5xx / timeout / network / empty body) retry the same model once,
then try the fallback model. 400/404/422 skip straight to the fallback. 401/403 fail immediately.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 502, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.status, self.retryable = code, message, status, retryable


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    retries: int
    used_fallback: bool
    ms: int


class _Attempt(Exception):
    def __init__(self, kind: str, err: LLMError, retry_after: float | None = None):
        self.kind, self.err, self.retry_after = kind, err, retry_after   # kind: retry | switch | fatal


class LLMClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 45.0,
                 transport: httpx.AsyncBaseTransport | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.api_key = api_key
        self._sleep = sleep
        self._c = httpx.AsyncClient(base_url=base_url, transport=transport,
                                    timeout=httpx.Timeout(timeout_s, connect=6.0))

    async def _once(self, model: str, messages: list[dict], max_tokens: int, temperature: float):
        try:
            r = await self._c.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature, "stream": False},
            )
        except httpx.TimeoutException:
            raise _Attempt("retry", LLMError("LLM_TIMEOUT", "The language model took too long to respond.",
                                             status=504, retryable=True))
        except httpx.TransportError as e:
            raise _Attempt("retry", LLMError("LLM_UNAVAILABLE", f"Cannot reach the language model ({type(e).__name__}).",
                                             status=502, retryable=True))
        sc = r.status_code
        if sc == 429:
            ra = r.headers.get("retry-after")
            try:
                ra_f = float(ra) if ra else None
            except ValueError:
                ra_f = None
            raise _Attempt("retry", LLMError("LLM_RATE_LIMITED", "The language model is rate-limited right now.",
                                             status=429, retryable=True), ra_f)
        if sc >= 500:
            raise _Attempt("retry", LLMError("LLM_UNAVAILABLE", f"Language model provider error ({sc}).",
                                             status=502, retryable=True))
        if sc in (401, 403):
            raise _Attempt("fatal", LLMError("LLM_UNAVAILABLE", "Authentication with the language model provider failed.",
                                             status=502, retryable=False))
        if sc >= 400:
            raise _Attempt("switch", LLMError("LLM_UNAVAILABLE", f"Language model request rejected ({sc}); check the model id.",
                                              status=502, retryable=False))
        try:
            d = r.json()
            choice = d["choices"][0]
            text = _THINK.sub("", choice["message"]["content"] or "").strip()
            usage = d.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Attempt("retry", LLMError("LLM_UNAVAILABLE", "Malformed response from the language model.",
                                             status=502, retryable=True))
        if not text:
            raise _Attempt("retry", LLMError("LLM_UNAVAILABLE", "The language model returned an empty answer.",
                                             status=502, retryable=True))
        return text, usage, choice.get("finish_reason")

    async def chat(self, *, model: str, messages: list[dict], max_tokens: int,
                   temperature: float = 0.2, fallback_model: str = "") -> LLMResult:
        if not self.api_key or not model:
            raise LLMError("LLM_UNAVAILABLE", "The language model isn't configured on the server "
                           "(set LLM_API_KEY and LLM_CHAT_MODEL).", status=503)
        t0 = time.perf_counter()
        queue = [model, model] + ([fallback_model] if fallback_model and fallback_model != model else [])
        last: _Attempt | None = None
        retries = 0
        while queue:
            m = queue.pop(0)
            try:
                text, usage, fin = await self._once(m, messages, max_tokens, temperature)
                return LLMResult(text, m, usage.get("prompt_tokens"), usage.get("completion_tokens"), fin,
                                 retries, m != model, round((time.perf_counter() - t0) * 1000))
            except _Attempt as a:
                last = a
                if a.kind == "fatal":
                    raise a.err
                if a.kind == "switch":
                    queue = [x for x in queue if x != model]
                    continue
                retries += 1
                if queue:
                    await self._sleep(min(a.retry_after if a.retry_after is not None else 0.6 * retries, 5.0))
        raise last.err  # type: ignore[union-attr]

    async def aclose(self) -> None:
        await self._c.aclose()
