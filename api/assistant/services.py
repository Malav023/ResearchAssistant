from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from config import AppConfig
from llm.client import LLMClient
from ml.runtime import MLRuntime
from papers_service import search_papers
from ra_core.cache import RedisCache, SessionStore
from ra_core.rag import HashingEmbedder
from ra_core.rag.embedder import Embedder
from ra_core.rag.types import Paper

from .ratelimit import RateLimiter
from .repo import SessionRepo
from .tracing import TraceStore

log = logging.getLogger("assistant")


@dataclass
class Services:
    cfg: AppConfig
    ml: MLRuntime
    embedder: Embedder
    llm: LLMClient
    cache: RedisCache
    sessions: SessionStore
    repo: SessionRepo
    traces: TraceStore
    fetch_papers: Callable[[str, int], Awaitable[list[Paper]]]
    limiter: RateLimiter

    async def aclose(self) -> None:
        for closer in (self.ml.aclose, self.llm.aclose, self.cache.aclose):
            try:
                await closer()
            except Exception:
                log.exception("error while closing")


def make_ml(cfg: AppConfig) -> MLRuntime:
    if cfg.ml_backend == "remote":
        from ml.remote import RemoteMLRuntime
        return RemoteMLRuntime(cfg.ml_remote_url, cfg.ml_service_token, cfg.ml_remote_timeout_s)
    if cfg.ml_backend == "local":
        from ml.local import LocalMLRuntime
        return LocalMLRuntime()
    if cfg.ml_backend == "stub":
        from ml.stub import StubMLRuntime
        return StubMLRuntime()
    raise ValueError(f"ML_BACKEND must be stub | remote | local, got {cfg.ml_backend!r}")


def make_embedder(cfg: AppConfig) -> Embedder:
    if cfg.embed_backend == "remote":
        from ml.remote import RemoteEmbedder
        return RemoteEmbedder(cfg.ml_remote_url, cfg.ml_service_token, cfg.ra.embed_model, cfg.embed_timeout_s)
    if cfg.embed_backend == "local":
        from ra_core.rag.embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(cfg.ra.embed_model)
    if cfg.embed_backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"EMBED_BACKEND must be hashing | remote | local, got {cfg.embed_backend!r}")


def build_services(cfg: AppConfig, db) -> Services:
    cache = RedisCache(cfg.ra.redis_url, timeout_s=cfg.ra.redis_timeout_s,
                       breaker_threshold=cfg.ra.breaker_threshold, breaker_cooldown_s=cfg.ra.breaker_cooldown_s)

    async def fetch(topic: str, limit: int) -> list[Paper]:
        return await search_papers(topic, limit=limit, openalex_key=cfg.openalex_key, s2_key=cfg.s2_key)

    return Services(
        cfg=cfg, ml=make_ml(cfg), embedder=make_embedder(cfg),
        llm=LLMClient(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, timeout_s=cfg.llm_timeout_s),
        cache=cache, sessions=SessionStore(cache, cfg.ra), repo=SessionRepo(db),
        traces=TraceStore(db, cfg.trace_ttl_days), fetch_papers=fetch,
        limiter=RateLimiter(cfg.rate_limit_per_min),
    )
