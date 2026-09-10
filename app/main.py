"""FastAPI application factory.

Layering, top to bottom:

    api/routes  → HTTP only: parse, delegate, present
    services/   → the logic: vision pipeline, matching, guard, jobs, costs
    repositories/ → tenant-scoped data access
    db/, providers/ → persistence and external AI

A route never touches the ORM directly and a service never imports FastAPI.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.routes import costs, evaluation, health, images, jobs, posts, review
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, safe_extra
from app.worker import WorkerLoop, reclaim_stale_jobs

logger = logging.getLogger("app.http")

DESCRIPTION = """
Understands an image library, organises it automatically, and matches the right
image to the right article — **with a mismatch guard that refuses rather than
guesses**.

* `POST /v1/jobs` runs vision tagging, embedding and matching as background work.
* `GET /v1/posts/{slug}/images` returns ranked, guarded suggestions — or an
  explained "no confident match".
* `POST /v1/posts/{slug}/images/{image_id}/check` forces one pairing through
  the guard.
* `GET /v1/costs` is the per-call AI cost ledger.
"""


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        worker: Optional[WorkerLoop] = None
        if settings.worker_enabled:
            # Requeue anything a previous process died holding, then start the
            # in-process worker so `uvicorn app.main:app` is a complete system.
            reclaim_stale_jobs(settings)
            worker = WorkerLoop(settings)
            worker.start()
        application.state.worker = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    application = FastAPI(
        title="FlyRank — AI Image Understanding & Content Matching Engine",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    _register_middleware(application)
    _register_exception_handlers(application)

    application.include_router(health.router)
    for module in (images, posts, jobs, review, costs, evaluation):
        application.include_router(module.router, prefix=settings.api_prefix)

    return application


def _register_middleware(application: FastAPI) -> None:
    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _register_exception_handlers(application: FastAPI) -> None:
    """Bad input becomes a clean 4xx; only genuine bugs become a 500."""

    @application.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.info(
            "app_error",
            extra=safe_extra({
                "code": exc.code,
                "status": exc.status_code,
                "path": request.url.path,
                "request_id": getattr(request.state, "request_id", None),
            }),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Request body or parameters failed validation.",
                "details": {"errors": _clean_errors(exc.errors())},
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": _CODE_BY_STATUS.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
                "details": {},
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "unhandled_error",
            extra=safe_extra({"path": request.url.path, "request_id": request_id}),
        )
        # Never leak an internal message or stack trace to the client.
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "details": {"request_id": request_id},
            },
        )


_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    429: "too_many_requests",
}


def _clean_errors(errors) -> list:
    """Drop the non-serialisable ``ctx`` payload pydantic attaches."""
    cleaned = []
    for error in errors:
        cleaned.append(
            {
                "loc": [str(part) for part in error.get("loc", [])],
                "type": error.get("type"),
                "msg": error.get("msg"),
            }
        )
    return cleaned


app = create_app()
