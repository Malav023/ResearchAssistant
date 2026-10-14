"""Model server (runs on Colab). Wraps ml.engine.Engine behind a tiny authenticated HTTP API.

    uvicorn ml_service:app --host 127.0.0.1 --port 8000      # env: ML_SERVICE_TOKEN, HF_USER, HF_TOKEN, ...

POST /classify {text}                -> {label, confidence, probs, version, ms}
POST /extract  {text, intent?}       -> {topic, avg_logprob, version, ms}
POST /embed    {texts, kind}         -> {model, vectors}        (kind: passage | query)
GET  /health                         -> {ok, ...}               (no auth, no secrets)
"""
import hmac
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ml.engine import Engine

TOKEN = os.environ.get("ML_SERVICE_TOKEN", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
SERVE_EMBEDDINGS = os.environ.get("SERVE_EMBEDDINGS", "true").lower() in ("1", "true", "yes")

engine = Engine()
state = {"embedder": None, "loaded_s": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    try:
        engine.load()
        if SERVE_EMBEDDINGS:
            from ra_core.rag.embedder import SentenceTransformerEmbedder
            state["embedder"] = SentenceTransformerEmbedder(EMBED_MODEL, device=os.environ.get("EMBED_DEVICE", "cpu"))
        engine.warmup()
        state["loaded_s"] = round(time.time() - t, 1)
    except Exception as e:                     # keep the server up so /health can report the reason
        state["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="Research Assistant model server", lifespan=lifespan)


def auth(authorization: str = Header(default="")):
    if not TOKEN:
        raise HTTPException(500, "ML_SERVICE_TOKEN is not set on the server")
    if not hmac.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(401, "bad token")


class ClassifyIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ExtractIn(ClassifyIn):
    intent: str | None = None


class EmbedIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    kind: str = Field(default="passage", pattern="^(passage|query)$")


@app.get("/health")
def health():
    return {"ok": engine.ready and state["error"] is None, "error": state["error"], "loaded_s": state["loaded_s"],
            "embeddings": state["embedder"] is not None, "embed_model": EMBED_MODEL, **engine.info}


def _need_ready():
    if not engine.ready:
        raise HTTPException(503, state["error"] or "models still loading")


@app.post("/classify", dependencies=[Depends(auth)])
def classify(body: ClassifyIn):
    _need_ready()
    t = time.perf_counter()
    label, conf, probs = engine.classify(body.text)
    return {"label": label, "confidence": conf, "probs": probs, "version": engine.CLF_VERSION,
            "ms": round((time.perf_counter() - t) * 1000)}


@app.post("/extract", dependencies=[Depends(auth)])
def extract(body: ExtractIn):
    _need_ready()
    t = time.perf_counter()
    topic, lp = engine.extract_topic(body.text)
    return {"topic": topic, "avg_logprob": lp, "version": engine.EXT_VERSION,
            "ms": round((time.perf_counter() - t) * 1000)}


@app.post("/embed", dependencies=[Depends(auth)])
def embed(body: EmbedIn):
    emb = state["embedder"]
    if emb is None:
        raise HTTPException(503, "embeddings not enabled on this server")
    vecs = emb.encode_passages(body.texts) if body.kind == "passage" else emb._enc([emb._q_prefix + t for t in body.texts])
    return {"model": EMBED_MODEL, "vectors": vecs.tolist()}
