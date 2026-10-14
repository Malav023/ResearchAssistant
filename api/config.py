"""The ONLY place that reads the environment.

dotenv is loaded here, before anything else reads os.environ. (Previously main.py imported the
routers first, so auth_utils/db/papers_routes read env vars BEFORE load_dotenv() ran.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from ra_core.settings import Settings, load_settings  # noqa: E402  (after dotenv on purpose)

_PLACEHOLDERS = ("ignore", "changeme", "change-this", "your_", "xxx")


def _s(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _key(name: str) -> str:
    """API keys: '' and placeholder strings like 'ignoreWritten' count as 'not set'."""
    v = _s(name)
    return "" if (not v or v.lower().startswith(_PLACEHOLDERS)) else v


def _b(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    return int(_s(name, str(default)))


def _f(name: str, default: float) -> float:
    return float(_s(name, str(default)))


@dataclass(frozen=True)
class AppConfig:
    # --- existing app ---
    mongo_uri: str
    mongo_db: str
    jwt_secret: str
    jwt_expire_minutes: int
    frontend_origin: str
    cookie_secure: bool
    openalex_key: str
    s2_key: str
    # --- ML (Colab model server or in-process) ---
    ml_backend: str            # stub | remote | local
    ml_remote_url: str
    ml_service_token: str
    ml_remote_timeout_s: float
    classifier_timeout_s: float
    extractor_timeout_s: float
    ml_warmup: bool
    embed_backend: str         # hashing | remote | local
    embed_timeout_s: float
    intent_min_conf: float
    extract_on_chat: bool
    # --- chat/report LLM API (OpenAI-compatible) ---
    llm_base_url: str
    llm_api_key: str
    llm_chat_model: str
    llm_report_model: str
    llm_fallback_model: str
    llm_timeout_s: float
    llm_max_tokens_chat: int
    llm_max_tokens_report: int
    # --- pipeline ---
    rag_mode: str              # hybrid | stuff
    sources_limit: int
    history_turns: int
    rate_limit_per_min: int
    # --- observability ---
    debug: bool
    trace_store_text: bool
    trace_ttl_days: int
    ra: Settings


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    ml_backend = _s("ML_BACKEND", "stub").lower()
    chat_model = _s("LLM_CHAT_MODEL")
    return AppConfig(
        mongo_uri=_s("MONGO_URI", "mongodb://localhost:27017"),
        mongo_db=_s("MONGO_DB_NAME", "research_assistant"),
        jwt_secret=_s("JWT_SECRET", "dev-secret-change-me"),
        jwt_expire_minutes=_i("JWT_EXPIRE_MINUTES", 60),
        frontend_origin=_s("FRONTEND_ORIGIN", "http://localhost:5173"),
        cookie_secure=_b("COOKIE_SECURE", False),
        openalex_key=_key("OPENALEX_API_KEY"),
        s2_key=_key("S2_API_KEY"),
        ml_backend=ml_backend,
        ml_remote_url=_s("ML_REMOTE_URL").rstrip("/"),
        ml_service_token=_s("ML_SERVICE_TOKEN"),
        ml_remote_timeout_s=_f("ML_REMOTE_TIMEOUT_S", 25.0),
        classifier_timeout_s=_f("CLASSIFIER_TIMEOUT_S", 8.0),
        extractor_timeout_s=_f("EXTRACTOR_TIMEOUT_S", 20.0),
        ml_warmup=_b("ML_WARMUP", True),
        embed_backend=_s("EMBED_BACKEND", {"stub": "hashing", "remote": "remote", "local": "local"}.get(ml_backend, "hashing")).lower(),
        embed_timeout_s=_f("EMBED_TIMEOUT_S", 15.0),
        intent_min_conf=_f("INTENT_MIN_CONF", 0.6),
        extract_on_chat=_b("EXTRACT_ON_CHAT", False),
        llm_base_url=_s("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        llm_api_key=_key("LLM_API_KEY"),
        llm_chat_model=chat_model,
        llm_report_model=_s("LLM_REPORT_MODEL") or chat_model,
        llm_fallback_model=_s("LLM_FALLBACK_MODEL"),
        llm_timeout_s=_f("LLM_TIMEOUT_S", 45.0),
        llm_max_tokens_chat=_i("LLM_MAX_TOKENS_CHAT", 600),
        llm_max_tokens_report=_i("LLM_MAX_TOKENS_REPORT", 1500),
        rag_mode=_s("RAG_MODE", "hybrid").lower(),
        sources_limit=_i("SOURCES_LIMIT", 5),
        history_turns=_i("HISTORY_TURNS", 6),
        rate_limit_per_min=_i("RATE_LIMIT_PER_MIN", 20),
        debug=_b("DEBUG", False),
        trace_store_text=_b("TRACE_STORE_TEXT", True),
        trace_ttl_days=_i("TRACE_TTL_DAYS", 30),
        ra=load_settings(),
    )
