"""Liveness and readiness."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.schemas.api import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness + configuration")
def health(
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> HealthOut:
    try:
        session.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # surfaced, not raised: /health must stay cheap
        database = f"error: {exc}"
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        version=__version__,
        vision_provider=settings.vision_provider,
        embedding_provider=settings.embedding_provider,
        database=database,
    )
