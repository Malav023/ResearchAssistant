import asyncio
import dataclasses

import fakeredis.aioredis
import httpx
import pytest
from bson import ObjectId
from fastapi import FastAPI
from mongomock_motor import AsyncMongoMockClient

from assistant.errors import install_error_handlers
from assistant.ratelimit import RateLimiter
from assistant.repo import SessionRepo
from assistant.routes import router
from assistant.services import Services
from assistant.tracing import TraceStore, install_trace_middleware
from auth_routes import get_current_user
from config import get_config
from llm.client import LLMResult
from ml.runtime import INTENTS, ClassifyResult, ExtractResult
from ra_core.cache import RedisCache, SessionStore
from ra_core.rag import HashingEmbedder, Paper

TOPICS = [
    ("Privacy-Preserving Federated Learning at Scale", "We study federated learning under differential privacy. Clients share only noisy gradients."),
    ("Transformers versus State Space Models", "We compare transformer and state space architectures. The evaluation setup uses identical training budgets."),
    ("Catastrophic Forgetting in Large Language Models", "Language models forget earlier knowledge when fine-tuned on new tasks."),
    ("Secure Aggregation for Mobile Devices", "A protocol for secure aggregation of model updates from phones."),
    ("Continual Learning with Replay Buffers", "Replay buffers reduce forgetting in continual learning."),
]


def make_papers(n=5, abstracts=True):
    return [Paper(id=f"https://openalex.org/W{i}", title=t, abstract=a if abstracts else "", year=2020 + i,
                  first_author=f"Author {i}", citations=100 - i, top_concept="Machine learning", doi=f"10.1/{i}")
            for i, (t, a) in enumerate(TOPICS[:n], 1)]


class FakeML:
    name = "fake"

    def __init__(self):
        self.label, self.conf, self.topic = "fetch_sources", 0.93, "federated learning privacy"
        self.fail_classify = None
        self.fail_extract = None
        self.extract_delay = 0.0
        self.calls = {"classify": 0, "extract": 0}

    async def classify(self, text):
        self.calls["classify"] += 1
        if self.fail_classify:
            raise self.fail_classify
        rest = (1 - self.conf) / (len(INTENTS) - 1)
        return ClassifyResult(self.label, self.conf, {i: (self.conf if i == self.label else rest) for i in INTENTS},
                              version="fake")

    async def extract(self, text, intent):
        self.calls["extract"] += 1
        if self.extract_delay:
            await asyncio.sleep(self.extract_delay)
        if self.fail_extract:
            raise self.fail_extract
        return ExtractResult(topic=self.topic, avg_logprob=-0.2, version="fake")

    async def health(self):
        return {"ok": True, "backend": "fake"}

    async def warmup(self):
        pass

    async def aclose(self):
        pass


class FakeLLM:
    def __init__(self):
        self.reply = "It uses an identical budget [Paper 2]."
        self.error = None
        self.calls = []

    async def chat(self, *, model, messages, max_tokens, temperature=0.2, fallback_model=""):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        return LLMResult(self.reply, model, 10, 5, "stop", 0, False, 5)

    async def aclose(self):
        pass


class Env:
    pass


@pytest.fixture
async def env():
    e = Env()
    e.cfg = dataclasses.replace(get_config(), llm_api_key="k", llm_chat_model="chat-m", llm_report_model="rep-m",
                                debug=True, rate_limit_per_min=1000, intent_min_conf=0.6, extract_on_chat=False,
                                rag_mode="hybrid")
    e.db = AsyncMongoMockClient()["test"]
    e.ml, e.llm = FakeML(), FakeLLM()
    e.fetch_calls = []
    e.papers = make_papers()

    async def fetch(topic, limit):
        e.fetch_calls.append(topic)
        if getattr(e, "fetch_error", None):
            raise e.fetch_error
        return e.papers[:limit]

    cache = RedisCache(client=fakeredis.aioredis.FakeRedis())
    e.svc = Services(cfg=e.cfg, ml=e.ml, embedder=HashingEmbedder(), llm=e.llm, cache=cache,
                     sessions=SessionStore(cache, e.cfg.ra), repo=SessionRepo(e.db), traces=TraceStore(e.db),
                     fetch_papers=fetch, limiter=RateLimiter(1000))
    e.user = {"_id": ObjectId(), "name": "T", "email": "t@example.com"}
    app = FastAPI()
    app.state.svc = e.svc
    install_trace_middleware(app)
    install_error_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: e.user
    e.app = app
    e.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def post(msg, sid=None):
        return await e.client.post("/api/assistant/message", json={"message": msg, **({"session_id": sid} if sid else {})})

    e.post = post
    yield e
    await e.client.aclose()


@pytest.fixture
def papers():
    """3 short papers used by the ra_core RAG/cache tests."""
    return [
        Paper("a1", "Privacy-Preserving Federated Learning at Scale",
              "We study federated learning under differential privacy. Clients train locally and share only noisy gradients. "
              "Our method reduces communication cost by 40 percent. Experiments cover image and text tasks.",
              "Federated learning lets many devices train a shared model without centralising data. "
              "However, gradients can leak private information, e.g. via inversion attacks (Zhu et al. 2019). "
              "We therefore add calibrated noise and secure aggregation."),
        Paper("b2", "Transformers versus State Space Models: An Evaluation",
              "We compare transformer and state space model architectures on long-context benchmarks. "
              "The evaluation setup uses identical training budgets and the same tokenizer. "
              "State space models win on throughput while transformers win on recall.",
              "Attention scales quadratically with sequence length. State space models offer linear scaling. "
              "Prior comparisons used inconsistent evaluation protocols, which makes conclusions unreliable."),
        Paper("c3", "Catastrophic Forgetting in Large Language Models",
              "Large language models forget previously learned knowledge when fine-tuned on new tasks. "
              "We measure this catastrophic forgetting across model sizes. Replay buffers and regularisation reduce the effect.",
              "Continual learning aims to accumulate knowledge over time. Fine-tuning on a narrow dataset overwrites weights "
              "that encoded earlier skills."),
    ]
