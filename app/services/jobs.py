"""Background job orchestration.

Slow, bulk AI work never runs on the request path (shared requirement #3). A
request enqueues a job and returns immediately; the worker claims it, runs it,
and records progress, retries and — on final failure — an alert.

Every handler is written to be **resumable**: it processes only the items still
outstanding, so attempt 2 of a job that died half way does not re-pay for the
images attempt 1 already tagged.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import AppError, BudgetExceededError, ConflictError, ProviderError
from app.db import models
from app.db.base import utcnow
from app.providers.registry import build_embedding_provider, build_vision_provider
from app.repositories.embeddings import OWNER_IMAGE, OWNER_POST, EmbeddingRepository
from app.repositories.images import ImageRepository
from app.repositories.jobs import JobRepository
from app.repositories.posts import PostRepository
from app.services.alerts import send_job_failure_alert
from app.services.costs import CostTracker
from app.services.embedding_service import EmbeddingService
from app.services.matching import MatchingService
from app.core.logging import safe_extra

logger = logging.getLogger("app.jobs")

KIND_VISION_TAGGING = "vision_tagging"
KIND_EMBED_IMAGES = "embed_images"
KIND_EMBED_POSTS = "embed_posts"
KIND_MATCH_POSTS = "match_posts"
KIND_FULL_PIPELINE = "full_pipeline"

JOB_KINDS = (
    KIND_VISION_TAGGING,
    KIND_EMBED_IMAGES,
    KIND_EMBED_POSTS,
    KIND_MATCH_POSTS,
    KIND_FULL_PIPELINE,
)

#: The chain ``full_pipeline`` expands into, in order.
PIPELINE_STEPS = (
    KIND_VISION_TAGGING,
    KIND_EMBED_IMAGES,
    KIND_EMBED_POSTS,
    KIND_MATCH_POSTS,
)


@dataclass
class JobOutcome:
    processed: int = 0
    failed: int = 0
    details: Optional[dict] = None
    #: first non-fatal error seen; surfaced so a partly-failed job still retries
    error: Optional[str] = None


# --- enqueueing --------------------------------------------------------------
def enqueue_job(
    session: Session,
    tenant_id: str,
    settings: Settings,
    *,
    kind: str,
    dedupe_key: Optional[str] = None,
    payload: Optional[dict] = None,
) -> tuple:
    """Enqueue a job idempotently.

    Returns ``(job, created)``. A repeated submission with the same
    ``dedupe_key`` returns the original job instead of queueing a second copy —
    the retried action happens once (shared requirement #5).
    """
    if kind not in JOB_KINDS:
        raise AppError(
            f"unknown job kind {kind!r}; expected one of {', '.join(JOB_KINDS)}"
        )
    repo = JobRepository(session, tenant_id)
    key = dedupe_key or f"{kind}:{utcnow().strftime('%Y%m%dT%H%M%S%f')}"

    existing = repo.get_by_dedupe_key(key)
    if existing is not None:
        return existing, False

    job = repo.enqueue(
        kind=kind,
        dedupe_key=key,
        payload=payload or {},
        max_attempts=settings.job_max_attempts,
    )
    session.flush()
    logger.info(
        "job_enqueued",
        extra=safe_extra(
            {"job_id": job.id, "kind": kind, "tenant_id": tenant_id}
        ),
    )
    return job, True


# --- handlers ----------------------------------------------------------------
def handle_vision_tagging(
    session: Session, job: models.Job, settings: Settings
) -> JobOutcome:
    from app.services.vision_pipeline import VisionService

    images_repo = ImageRepository(session, job.tenant_id)
    pending = images_repo.list_pending(limit=job.payload.get("limit"))
    job.total_items = len(pending)
    session.flush()

    if not pending:
        return JobOutcome(details={"message": "no images pending classification"})

    service = VisionService(
        session,
        job.tenant_id,
        settings,
        build_vision_provider(settings),
        CostTracker(session, job.tenant_id, settings),
    )

    outcome = JobOutcome()
    flagged = 0
    for image in pending:
        try:
            tagged = service.tag_image(image, job_id=job.id)
            outcome.processed += 1
            flagged += int(bool(tagged.needs_review))
        except BudgetExceededError:
            # Stop the whole batch: continuing would just fail every remaining
            # image and fill the ledger with noise.
            raise
        except ProviderError as exc:
            outcome.failed += 1
            outcome.error = outcome.error or str(exc)
            logger.warning(
                "image_tagging_failed",
                extra=safe_extra(
                    {"image_id": image.id, "job_id": job.id, "error": str(exc)}
                ),
            )
        job.processed_items = outcome.processed
        job.failed_items = outcome.failed
        session.flush()

    outcome.details = {
        "tagged": outcome.processed,
        "flagged_for_review": flagged,
        "failed": outcome.failed,
    }
    return outcome


def handle_embed_images(
    session: Session, job: models.Job, settings: Settings
) -> JobOutcome:
    service = _embedding_service(session, job.tenant_id, settings)
    images_repo = ImageRepository(session, job.tenant_id)
    embed_repo = EmbeddingRepository(session, job.tenant_id)

    tagged = images_repo.list_tagged()
    missing_ids = set(
        embed_repo.missing_owner_ids(
            OWNER_IMAGE, [img.id for img in tagged], service.model_name
        )
    )
    todo = [img for img in tagged if img.id in missing_ids]

    job.total_items = len(todo)
    session.flush()

    outcome = JobOutcome()
    for image in todo:
        try:
            service.embed_image(image, job_id=job.id)
            outcome.processed += 1
        except BudgetExceededError:
            raise
        except ProviderError as exc:
            outcome.failed += 1
            outcome.error = outcome.error or str(exc)
        job.processed_items = outcome.processed
        job.failed_items = outcome.failed
        session.flush()

    outcome.details = {
        "embedded": outcome.processed,
        "already_embedded": len(tagged) - len(todo),
        "model": service.model_name,
    }
    return outcome


def handle_embed_posts(
    session: Session, job: models.Job, settings: Settings
) -> JobOutcome:
    service = _embedding_service(session, job.tenant_id, settings)
    posts_repo = PostRepository(session, job.tenant_id)
    embed_repo = EmbeddingRepository(session, job.tenant_id)

    posts = posts_repo.list_all()
    missing_ids = set(
        embed_repo.missing_owner_ids(
            OWNER_POST, [p.id for p in posts], service.model_name
        )
    )
    todo = [p for p in posts if p.id in missing_ids]

    job.total_items = len(todo)
    session.flush()

    outcome = JobOutcome()
    for post in todo:
        try:
            service.embed_post(post, job_id=job.id)
            outcome.processed += 1
        except BudgetExceededError:
            raise
        except ProviderError as exc:
            outcome.failed += 1
            outcome.error = outcome.error or str(exc)
        job.processed_items = outcome.processed
        job.failed_items = outcome.failed
        session.flush()

    outcome.details = {
        "embedded": outcome.processed,
        "already_embedded": len(posts) - len(todo),
        "model": service.model_name,
    }
    return outcome


def handle_match_posts(
    session: Session, job: models.Job, settings: Settings
) -> JobOutcome:
    service = _embedding_service(session, job.tenant_id, settings)
    matching = MatchingService(session, job.tenant_id, settings, service)
    posts = PostRepository(session, job.tenant_id).list_all()

    job.total_items = len(posts)
    session.flush()

    outcome = JobOutcome()
    with_match = 0
    for post in posts:
        result = matching.match_and_persist(post)
        with_match += int(result.has_match)
        outcome.processed += 1
        job.processed_items = outcome.processed
        session.flush()

    outcome.details = {
        "posts_matched": outcome.processed,
        "posts_with_confident_match": with_match,
        "posts_without_match": outcome.processed - with_match,
    }
    return outcome


def handle_full_pipeline(
    session: Session, job: models.Job, settings: Settings
) -> JobOutcome:
    """Run the whole chain in one job so a demo is a single call."""
    outcome = JobOutcome()
    details: Dict[str, dict] = {}
    for step in PIPELINE_STEPS:
        step_outcome = HANDLERS[step](session, job, settings)
        details[step] = step_outcome.details or {}
        outcome.processed += step_outcome.processed
        outcome.failed += step_outcome.failed
        outcome.error = outcome.error or step_outcome.error
    job.total_items = outcome.processed + outcome.failed
    job.processed_items = outcome.processed
    job.failed_items = outcome.failed
    outcome.details = details
    return outcome


HANDLERS: Dict[str, Callable[[Session, models.Job, Settings], JobOutcome]] = {
    KIND_VISION_TAGGING: handle_vision_tagging,
    KIND_EMBED_IMAGES: handle_embed_images,
    KIND_EMBED_POSTS: handle_embed_posts,
    KIND_MATCH_POSTS: handle_match_posts,
    KIND_FULL_PIPELINE: handle_full_pipeline,
}


# --- execution ---------------------------------------------------------------
def run_job(session: Session, job: models.Job, settings: Settings) -> models.Job:
    """Execute one claimed job, applying the retry / alert policy.

    The caller owns the transaction. On a retryable failure the job goes back
    to ``queued`` with exponential backoff; on the last attempt it is marked
    ``failed`` and an alert is raised.
    """
    handler = HANDLERS.get(job.kind)
    if handler is None:
        _fail_permanently(job, f"no handler registered for kind {job.kind!r}", settings)
        return job

    try:
        outcome = handler(session, job, settings)
    except Exception as exc:  # noqa: BLE001 - every failure mode is handled below
        retryable = getattr(exc, "retryable", True) and not isinstance(
            exc, (ConflictError, BudgetExceededError)
        )
        _handle_failure(job, exc, settings, retryable=retryable)
        return job

    if outcome.failed and outcome.error:
        # Some items failed. Retry the job — handlers are resumable, so the
        # next attempt only re-tries what is still outstanding.
        _handle_failure(
            job,
            ProviderError(
                f"{outcome.failed} item(s) failed: {outcome.error}"
            ),
            settings,
            retryable=True,
            partial=outcome,
        )
        return job

    job.status = models.JOB_SUCCEEDED
    job.finished_at = utcnow()
    job.result = outcome.details or {}
    job.last_error = None
    logger.info(
        "job_succeeded",
        extra=safe_extra({
            "job_id": job.id,
            "kind": job.kind,
            "attempts": job.attempts,
            "processed": outcome.processed,
        }),
    )
    return job


def _handle_failure(
    job: models.Job,
    exc: Exception,
    settings: Settings,
    *,
    retryable: bool,
    partial: Optional[JobOutcome] = None,
) -> None:
    message = str(exc)
    job.last_error = message[:2000]
    if partial is not None:
        job.result = partial.details or {}

    if retryable and job.attempts < job.max_attempts:
        backoff = min(
            settings.job_backoff_max_s,
            settings.job_backoff_base_s * (2 ** (job.attempts - 1)),
        )
        job.status = models.JOB_QUEUED
        job.scheduled_at = utcnow() + timedelta(seconds=backoff)
        job.started_at = None
        logger.warning(
            "job_retry_scheduled",
            extra=safe_extra({
                "job_id": job.id,
                "kind": job.kind,
                "attempt": job.attempts,
                "max_attempts": job.max_attempts,
                "backoff_s": backoff,
                "error": message,
            }),
        )
        return

    _fail_permanently(job, message, settings)


def _fail_permanently(job: models.Job, message: str, settings: Settings) -> None:
    job.status = models.JOB_FAILED
    job.finished_at = utcnow()
    job.last_error = message[:2000]
    send_job_failure_alert(
        job_id=job.id,
        tenant_id=job.tenant_id,
        kind=job.kind,
        attempts=job.attempts,
        error=message,
        webhook_url=settings.alert_webhook_url,
        context={
            "processed_items": job.processed_items,
            "failed_items": job.failed_items,
            "total_items": job.total_items,
        },
    )


def _embedding_service(
    session: Session, tenant_id: str, settings: Settings
) -> EmbeddingService:
    return EmbeddingService(
        session,
        tenant_id,
        settings,
        build_embedding_provider(settings),
        CostTracker(session, tenant_id, settings),
    )
