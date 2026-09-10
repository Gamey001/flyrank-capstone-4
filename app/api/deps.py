"""FastAPI dependencies.

The important one is ``get_tenant_id``: every route depends on it, so there is
no way to reach a repository without a resolved tenant.
"""

from typing import Iterator

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.db.session import get_session_factory
from app.repositories.tenants import get_tenant
from app.services.costs import CostTracker
from app.services.embedding_service import EmbeddingService
from app.services.matching import MatchingService


def get_db() -> Iterator[Session]:
    """One transaction per request, committed on a clean return."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_app_settings() -> Settings:
    return get_settings()


def get_tenant_id(
    request: Request,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    x_tenant_id: str = Header(
        default="",
        alias="X-Tenant-ID",
        description="Tenant to operate on; defaults to DEFAULT_TENANT.",
    ),
) -> str:
    tenant_id = (x_tenant_id or settings.default_tenant).strip()
    if get_tenant(session, tenant_id) is None:
        # An unknown tenant is a client error, not an invitation to create one.
        raise NotFoundError(
            f"unknown tenant {tenant_id!r}; run the seed script or pass a valid "
            "X-Tenant-ID header",
            details={"tenant_id": tenant_id},
        )
    request.state.tenant_id = tenant_id
    return tenant_id


def get_embedding_service(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
) -> EmbeddingService:
    from app.providers.registry import build_embedding_provider

    return EmbeddingService(
        session,
        tenant_id,
        settings,
        build_embedding_provider(settings),
        CostTracker(session, tenant_id, settings),
    )


def get_matching_service(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> MatchingService:
    return MatchingService(session, tenant_id, settings, embeddings)


def get_cost_tracker(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
) -> CostTracker:
    return CostTracker(session, tenant_id, settings)
