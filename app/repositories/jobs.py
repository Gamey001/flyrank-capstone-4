"""Job queue persistence.

The queue is a table. At this scale that is the right call: no broker to run,
the job history is queryable with the rest of the data, and ``FOR UPDATE SKIP
LOCKED`` gives a correct multi-worker claim on Postgres.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select

from app.db import models
from app.db.base import utcnow
from app.repositories.base import TenantRepository


class JobRepository(TenantRepository[models.Job]):
    model = models.Job

    def get_by_dedupe_key(self, dedupe_key: str) -> Optional[models.Job]:
        return self.session.scalars(
            self._scoped().where(models.Job.dedupe_key == dedupe_key)
        ).first()

    def list_recent(
        self, limit: int = 50, kind: Optional[str] = None
    ) -> List[models.Job]:
        stmt = self._scoped().order_by(models.Job.created_at.desc()).limit(limit)
        if kind:
            stmt = stmt.where(models.Job.kind == kind)
        return list(self.session.scalars(stmt))

    def enqueue(
        self,
        *,
        kind: str,
        dedupe_key: str,
        payload: dict,
        max_attempts: int,
        delay_s: float = 0.0,
    ) -> models.Job:
        return self.add(
            models.Job(
                kind=kind,
                dedupe_key=dedupe_key,
                payload=payload,
                max_attempts=max_attempts,
                status=models.JOB_QUEUED,
                scheduled_at=utcnow() + timedelta(seconds=delay_s),
            )
        )


def claim_next_job(session, *, now: Optional[datetime] = None) -> Optional[models.Job]:
    """Atomically take one due job across all tenants.

    Deliberately *not* tenant-scoped: the worker serves every tenant. Isolation
    is preserved because the job carries its own ``tenant_id`` and the handler
    builds tenant-scoped repositories from it.
    """
    now = now or utcnow()
    stmt = (
        select(models.Job)
        .where(
            models.Job.status == models.JOB_QUEUED,
            models.Job.scheduled_at <= now,
        )
        .order_by(models.Job.scheduled_at, models.Job.created_at)
        .limit(1)
    )
    dialect = session.bind.dialect.name if session.bind else ""
    if dialect == "postgresql":
        # Lets several worker processes share the queue without double-running.
        stmt = stmt.with_for_update(skip_locked=True)

    job = session.scalars(stmt).first()
    if job is None:
        return None

    job.status = models.JOB_RUNNING
    job.started_at = now
    job.attempts += 1
    session.flush()
    return job


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands back naive datetimes; normalise before comparing."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
