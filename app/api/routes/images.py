"""Corpus endpoints: ingest, browse, inspect."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_tenant_id
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.repositories.images import ImageRepository
from app.schemas.api import ImageOut, IngestRequest, IngestResponse
from app.services.ingestion import ingest_directory

router = APIRouter(prefix="/images", tags=["images"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Register corpus files (idempotent on content hash)",
)
def ingest(
    payload: IngestRequest,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
) -> IngestResponse:
    summary = ingest_directory(
        session, tenant_id, payload.images_dir or settings.images_dir
    )
    return IngestResponse(**summary)


@router.get("", response_model=List[ImageOut], summary="List corpus images")
def list_images(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    status: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    needs_review: Optional[bool] = Query(
        default=None, description="Only images flagged for human review."
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[ImageOut]:
    repo = ImageRepository(session, tenant_id)
    rows = repo.list_filtered(
        status=status,
        category=category,
        needs_review=needs_review,
        limit=limit,
        offset=offset,
    )
    return [ImageOut.model_validate(row) for row in rows]


@router.get("/stats", summary="Corpus counts by status")
def image_stats(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    repo = ImageRepository(session, tenant_id)
    by_status = repo.counts_by_status()
    flagged = len(repo.list_filtered(needs_review=True, limit=500))
    return {
        "total": repo.count(),
        "by_status": by_status,
        "flagged_for_review": flagged,
    }


@router.get("/{image_id}", response_model=ImageOut, summary="Inspect one image")
def get_image(
    image_id: str,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> ImageOut:
    image = ImageRepository(session, tenant_id).get(image_id)
    if image is None:
        raise NotFoundError(f"image {image_id!r} not found")
    return ImageOut.model_validate(image)
