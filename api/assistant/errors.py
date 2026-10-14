from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("assistant")


class AppError(Exception):
    """User-safe, typed error. `message` is shown to the student; internals go to the trace."""

    def __init__(self, code: str, message: str, *, stage: str = "internal", status: int = 500,
                 retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.stage, self.status, self.retryable = code, message, stage, status, retryable


def error_body(err: AppError, trace_id: str) -> dict:
    return {"error": {"code": err.code, "stage": err.stage, "message": err.message,
                      "retryable": err.retryable, "trace_id": trace_id}}


def _tid(request: Request) -> str:
    return getattr(request.state, "trace_id", None) or uuid.uuid4().hex[:12]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        tid = _tid(request)
        return JSONResponse(error_body(exc, tid), status_code=exc.status, headers={"X-Trace-Id": tid})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/api/assistant"):
            tid = _tid(request)
            err = AppError("VALIDATION", "Your message couldn't be read (empty or too long?).",
                           stage="validate", status=422)
            return JSONResponse(error_body(err, tid), status_code=422, headers={"X-Trace-Id": tid})
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        tid = _tid(request)
        log.exception("unhandled error trace_id=%s", tid)
        err = AppError("INTERNAL", "Something went wrong on our side.", stage="internal")
        return JSONResponse(error_body(err, tid), status_code=500, headers={"X-Trace-Id": tid})
