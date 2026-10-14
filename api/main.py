import asyncio
import logging
from contextlib import asynccontextmanager

from config import get_config  # MUST stay first: loads .env before any other module reads it

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from assistant.errors import install_error_handlers
from assistant.routes import router as assistant_router
from assistant.services import build_services
from assistant.tracing import install_trace_middleware, setup_logging
from auth_routes import router as auth_router
from db import get_db
from graph_builder import build_graph, fetch_openalex_papers, fetch_s2_paper_details
from papers_routes import router as papers_router
from sessions_routes import router as sessions_router

cfg = get_config()
log = logging.getLogger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(cfg.debug)
    if cfg.jwt_secret == "dev-secret-change-me":
        log.warning("JWT_SECRET is the built-in default: set a long random value in api/.env")
    svc = build_services(cfg, get_db())
    app.state.svc = svc
    try:
        await svc.repo.ensure_indexes()
        await svc.traces.ensure_indexes()
    except Exception as e:
        log.warning("could not create Mongo indexes: %s", e)
    log.info("config: ml=%s embed=%s redis=%s llm_configured=%s rag=%s", cfg.ml_backend, cfg.embed_backend,
             "on" if svc.cache.enabled else "off", bool(cfg.llm_api_key and cfg.llm_chat_model), cfg.rag_mode)
    if cfg.ml_warmup:
        try:
            await asyncio.wait_for(svc.ml.warmup(), timeout=120)
        except Exception as e:                   # e.g. Colab not started yet: run degraded, don't crash
            log.warning("ML warmup failed (the app will use keyword fallbacks until it recovers): %s", e)
    yield
    await svc.aclose()


app = FastAPI(title="Research Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[cfg.frontend_origin],
    allow_credentials=True,                      # required so the auth cookie is sent/received
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)
install_trace_middleware(app)
install_error_handlers(app)

app.include_router(auth_router)
app.include_router(papers_router)
app.include_router(sessions_router)
app.include_router(assistant_router)


@app.get("/api/graph")
async def get_citation_graph(q: str = Query(..., min_length=2, description="Research topic or paper title")):
    """Full citation graph: only called when the user opens the Citation Graph modal."""
    try:
        oa_papers = await fetch_openalex_papers(q, api_key=cfg.openalex_key)
        if not oa_papers:
            return {"nodes": [], "edges": [], "query": q}
        dois = [f"DOI:{p['doi'].replace('https://doi.org/', '')}" for p in oa_papers if p.get("doi")]
        s2_data = []
        if dois:
            try:
                s2_data = await fetch_s2_paper_details(dois[:20], api_key=cfg.s2_key)
            except Exception as e:
                logging.getLogger("graph").warning("S2 fetch failed (non-fatal): %s", e)
        graph = build_graph(oa_papers, s2_data)
        graph["query"] = q
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
