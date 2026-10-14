from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Query, Request

from auth_routes import get_current_user
from ra_core.rag.types import Paper

from .errors import AppError
from .pipeline import _views, handle_message
from .schemas import FeedbackRequest, MessageRequest, MessageResponse
from .tracing import Trace

log = logging.getLogger("assistant")
router = APIRouter(prefix="/api", tags=["assistant"])


def _svc(request: Request):
    return request.app.state.svc


def _uid(user: dict) -> str:
    return str(user["_id"])


async def _emit(svc, trace: Trace) -> None:
    doc = trace.doc()
    c = doc.get("classifier", {})
    log.info("assistant_request", extra={"fields": {
        "intent": doc.get("intent"), "action": doc.get("action"), "p_top": c.get("p_top"), "gated": c.get("gated"),
        "degraded": doc["degraded"], "total_ms": doc["total_ms"], "stages_ms": doc["stages_ms"],
        "error": (doc["error"] or {}).get("code")}})
    try:
        await svc.traces.save(trace)
    except Exception:
        log.exception("could not store trace")


@router.post("/assistant/message", response_model=MessageResponse)
async def message(body: MessageRequest, request: Request, user=Depends(get_current_user)):
    svc = _svc(request)
    trace = Trace(getattr(request.state, "trace_id", uuid.uuid4().hex[:12]), _uid(user))
    try:
        resp = await handle_message(svc, user, body, trace)
    except AppError as e:
        trace.fail(e)
        raise
    except Exception as e:
        log.exception("assistant pipeline crashed")
        err = AppError("INTERNAL", "Something went wrong on our side.", stage="internal")
        trace.fail(err, detail=f"{type(e).__name__}: {e}")
        raise err from e
    finally:
        await _emit(svc, trace)
    resp.degraded = list(trace.degraded)
    if svc.cfg.debug:
        resp.debug = trace.doc()
    return resp


@router.get("/assistant/sessions/{sid}")
async def session_detail(sid: str, request: Request, user=Depends(get_current_user)):
    svc = _svc(request)
    doc = await svc.repo.get_session(_uid(user), sid)
    if not doc:
        raise AppError("SESSION_NOT_FOUND", "That conversation no longer exists.", stage="session", status=404)
    papers = [Paper.from_dict(d) for d in doc.get("papers") or []]
    msgs = await svc.repo.get_messages(sid, limit=200)
    return {
        "session_id": sid, "title": doc.get("query", ""), "papers": [p.model_dump() for p in _views(papers)],
        "messages": [{"id": str(m["_id"]), "role": m["role"], "content": m["content"], "intent": m.get("intent"),
                      "action": m.get("action"), "citations": m.get("citations", []), "trace_id": m.get("trace_id"),
                      "ts": m["ts"].isoformat()} for m in msgs],
    }


@router.post("/assistant/feedback")
async def feedback(body: FeedbackRequest, request: Request, user=Depends(get_current_user)):
    ok = await _svc(request).traces.set_feedback(
        _uid(user), body.trace_id, {"correct_intent": body.correct_intent, "note": body.note})
    if not ok:
        raise AppError("NOT_FOUND", "Unknown trace.", stage="feedback", status=404)
    return {"ok": True}


@router.get("/health/deep")
async def health_deep(request: Request):
    svc, cfg = _svc(request), _svc(request).cfg
    try:
        await svc.repo.db.command("ping")
        mongo = True
    except Exception:
        mongo = False
    redis = "disabled" if not svc.cache.enabled else await svc.cache.ping()
    ml = await svc.ml.health()
    return {
        "ok": mongo, "mongo": mongo, "redis": redis, "ml": ml,
        "embedder": {"backend": cfg.embed_backend, "model": svc.embedder.name},
        "llm": {"configured": bool(cfg.llm_api_key and cfg.llm_chat_model), "chat_model": cfg.llm_chat_model,
                "report_model": cfg.llm_report_model, "fallback_model": cfg.llm_fallback_model or None},
        "settings": {"intent_min_conf": cfg.intent_min_conf, "extract_on_chat": cfg.extract_on_chat,
                     "rag_mode": cfg.rag_mode, "extractor_version": cfg.ra.extractor_version, "debug": cfg.debug},
    }


def _require_debug(request: Request) -> None:
    if not _svc(request).cfg.debug:
        raise AppError("NOT_FOUND", "Not found.", stage="debug", status=404)


@router.get("/debug/traces")
async def debug_traces(request: Request, user=Depends(get_current_user), intent: str | None = None,
                       max_conf: float | None = Query(default=None, ge=0, le=1), only_errors: bool = False,
                       limit: int = Query(default=50, ge=1, le=200)):
    _require_debug(request)
    return {"traces": await _svc(request).traces.list(_uid(user), intent=intent, max_conf=max_conf,
                                                      only_errors=only_errors, limit=limit)}


@router.get("/debug/traces/{trace_id}")
async def debug_trace(trace_id: str, request: Request, user=Depends(get_current_user)):
    _require_debug(request)
    d = await _svc(request).traces.get(_uid(user), trace_id)
    if not d:
        raise AppError("NOT_FOUND", "Unknown trace.", stage="debug", status=404)
    return d
