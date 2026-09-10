"""Job submission and progress."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_tenant_id
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.repositories.jobs import JobRepository
from app.schemas.api import JobCreate, JobOut
from app.services.jobs import enqueue_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue background AI work",
    response_description=(
        "202 for a newly queued job; 200 when an existing job with the same "
        "dedupe_key is returned instead."
    ),
)
def create_job(
    payload: JobCreate,
    response: Response,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
) -> JobOut:
    job, created = enqueue_job(
        session,
        tenant_id,
        settings,
        kind=payload.kind,
        dedupe_key=payload.dedupe_key,
        payload=payload.payload,
    )
    if not created:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replay"] = "true"
    return JobOut.model_validate(job)


@router.get("", response_model=List[JobOut], summary="Recent jobs")
def list_jobs(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    kind: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[JobOut]:
    rows = JobRepository(session, tenant_id).list_recent(limit=limit, kind=kind)
    return [JobOut.model_validate(row) for row in rows]


@router.get("/{job_id}", response_model=JobOut, summary="Job status and progress")
def get_job(
    job_id: str,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> JobOut:
    job = JobRepository(session, tenant_id).get(job_id)
    if job is None:
        raise NotFoundError(f"job {job_id!r} not found")
    return JobOut.model_validate(job)
