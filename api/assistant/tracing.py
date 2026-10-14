from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
log = logging.getLogger("assistant")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "level": record.levelname,
               "logger": record.name, "msg": record.getMessage(), "trace_id": trace_id_var.get()}
        out.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def setup_logging(debug: bool = False) -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_trace_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _trace(request: Request, call_next):
        tid = uuid.uuid4().hex[:12]
        request.state.trace_id = tid
        token = trace_id_var.set(tid)
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)
        response.headers["X-Trace-Id"] = tid
        return response


class Trace:
    """Everything we want to know about one assistant request (see BLUEPRINT section 6)."""

    def __init__(self, trace_id: str, user_id: str):
        self.trace_id, self.user_id = trace_id, user_id
        self.ts = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self.data: dict = {"stages_ms": {}}
        self.degraded: list[str] = []
        self.error: dict | None = None

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.data["stages_ms"][name] = round((time.perf_counter() - t) * 1000)

    def put(self, section: str, **kv) -> None:
        self.data.setdefault(section, {}).update(kv)

    def set(self, **kv) -> None:
        self.data.update(kv)

    def degrade(self, flag: str) -> None:
        if flag not in self.degraded:
            self.degraded.append(flag)

    def fail(self, err, detail: str | None = None) -> None:
        self.error = {"code": err.code, "stage": err.stage, "message": err.message, "detail": detail}

    def doc(self) -> dict:
        return {"trace_id": self.trace_id, "user_id": self.user_id, "ts": self.ts.isoformat(), **self.data,
                "degraded": list(self.degraded), "error": self.error,
                "total_ms": round((time.perf_counter() - self._t0) * 1000)}


class TraceStore:
    def __init__(self, db, ttl_days: int = 30):
        self.col = db["assistant_traces"]
        self.ttl_days = ttl_days

    async def ensure_indexes(self) -> None:
        await self.col.create_index("ts", expireAfterSeconds=self.ttl_days * 86400)
        await self.col.create_index("trace_id")
        await self.col.create_index([("user_id", 1), ("ts", -1)])

    async def save(self, trace: Trace) -> None:
        d = trace.doc()
        d["ts"] = trace.ts
        await self.col.insert_one(d)

    @staticmethod
    def _clean(d: dict) -> dict:
        d = {k: v for k, v in d.items() if k != "_id"}
        if isinstance(d.get("ts"), datetime):
            d["ts"] = d["ts"].isoformat()
        return d

    async def list(self, user_id: str, *, intent: str | None = None, max_conf: float | None = None,
                   only_errors: bool = False, limit: int = 50) -> list[dict]:
        q: dict = {"user_id": user_id}
        if intent:
            q["intent"] = intent
        if max_conf is not None:
            q["classifier.p_top"] = {"$lte": max_conf}
        if only_errors:
            q["error"] = {"$ne": None}
        cur = self.col.find(q).sort("ts", -1).limit(max(1, min(limit, 200)))
        return [self._clean(d) async for d in cur]

    async def get(self, user_id: str, trace_id: str) -> dict | None:
        d = await self.col.find_one({"user_id": user_id, "trace_id": trace_id})
        return self._clean(d) if d else None

    async def set_feedback(self, user_id: str, trace_id: str, fb: dict) -> bool:
        r = await self.col.update_one({"user_id": user_id, "trace_id": trace_id}, {"$set": {"feedback": fb}})
        return r.matched_count > 0
