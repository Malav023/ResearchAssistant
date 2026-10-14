from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- Redis ---
    redis_url: str = ""               # "" -> caching disabled (local dev / tests)
    redis_timeout_s: float = 0.3      # fail fast: a slow cache is worse than no cache
    breaker_threshold: int = 3        # consecutive failures before skipping Redis
    breaker_cooldown_s: float = 30.0
    # --- versions baked into cache keys (bump => old entries are ignored) ---
    extractor_version: str = "v1"
    # --- TTLs (seconds) ---
    ttl_extract_s: int = 7 * 24 * 3600
    ttl_sources_s: int = 6 * 3600     # "latest papers" => short
    ttl_session_s: int = 2 * 3600     # sliding
    # --- session / RAG ---
    history_max_msgs: int = 40
    local_sessions_max: int = 256
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rag_top_k: int = 5


def load_settings() -> Settings:
    e = os.getenv
    return Settings(
        redis_url=e("REDIS_URL", ""),
        redis_timeout_s=float(e("REDIS_TIMEOUT_S", "0.3")),
        extractor_version=e("EXTRACTOR_VERSION", "v1"),
        ttl_extract_s=int(e("TTL_EXTRACT_S", str(7 * 24 * 3600))),
        ttl_sources_s=int(e("TTL_SOURCES_S", str(6 * 3600))),
        ttl_session_s=int(e("TTL_SESSION_S", str(2 * 3600))),
        embed_model=e("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rag_top_k=int(e("RAG_TOP_K", "5")),
    )
