"""The background worker.

Two ways to run it:

* in-process, started by the API on startup (``WORKER_ENABLED=true``) — this is
  what makes ``docker compose up`` a single command and keeps the local demo to
  one terminal;
* standalone, ``python -m app.worker`` — what the compose file actually runs in
  production-shaped deployments, so slow AI work cannot starve the API.

Each iteration claims at most one job, runs it in its own transaction and
commits. A crash mid-job leaves the row in ``running``; the reaper below puts
it back on the queue on the next start.
"""

import logging
import signal
import threading
import time
from datetime import timedelta
from typing import Optional

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, safe_extra
from app.db import models
from app.db.base import utcnow
from app.db.session import session_scope
from app.repositories.jobs import as_utc, claim_next_job
from app.services.jobs import run_job

logger = logging.getLogger("app.worker")

#: A job still "running" after this long is assumed orphaned by a crash.
STALE_JOB_AFTER = timedelta(minutes=15)


def reclaim_stale_jobs(settings: Optional[Settings] = None) -> int:
    """Requeue jobs abandoned by a crashed worker."""
    cutoff = utcnow() - STALE_JOB_AFTER
    requeued = 0
    with session_scope() as session:
        rows = session.scalars(
            select(models.Job).where(models.Job.status == models.JOB_RUNNING)
        ).all()
        for job in rows:
            started = as_utc(job.started_at)
            if started is not None and started > cutoff:
                continue
            job.status = models.JOB_QUEUED
            job.scheduled_at = utcnow()
            job.started_at = None
            job.last_error = "requeued after worker restart"
            requeued += 1
    if requeued:
        logger.warning("stale_jobs_requeued", extra=safe_extra({"count": requeued}))
    return requeued


def run_once(settings: Optional[Settings] = None) -> Optional[str]:
    """Claim and run at most one due job. Returns its id, or None if idle."""
    settings = settings or get_settings()
    with session_scope() as session:
        job = claim_next_job(session)
        if job is None:
            return None
        job_id, kind = job.id, job.kind
        logger.info(
            "job_started",
            extra=safe_extra(
                {"job_id": job_id, "kind": kind, "attempt": job.attempts}
            ),
        )
        run_job(session, job, settings)
        return job_id


def drain(settings: Optional[Settings] = None, max_jobs: int = 100) -> int:
    """Run every currently-due job. Used by the seed script and the tests."""
    ran = 0
    while ran < max_jobs:
        if run_once(settings) is None:
            return ran
        ran += 1
    return ran


class WorkerLoop:
    """A polling worker on its own thread."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="flyrank-worker", daemon=True
        )
        self._thread.start()
        logger.info("worker_started", extra=safe_extra({"mode": "in-process"}))

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("worker_stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if run_once(self.settings) is None:
                    self._stop.wait(self.settings.worker_poll_interval_s)
            except Exception as exc:  # keep the loop alive; the job is already
                # marked failed/queued by run_job, this catches infrastructure
                # problems such as a dropped connection.
                logger.exception(
                    "worker_iteration_failed", extra=safe_extra({"error": str(exc)})
                )
                self._stop.wait(min(5.0, self.settings.worker_poll_interval_s * 10))


def main() -> None:  # pragma: no cover - process entry point
    settings = get_settings()
    configure_logging(settings.log_level)
    reclaim_stale_jobs(settings)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    logger.info("worker_started", extra=safe_extra({"mode": "standalone"}))
    while not stop.is_set():
        try:
            if run_once(settings) is None:
                time.sleep(settings.worker_poll_interval_s)
        except Exception as exc:
            logger.exception(
                "worker_iteration_failed", extra=safe_extra({"error": str(exc)})
            )
            time.sleep(2.0)
    logger.info("worker_stopped")


if __name__ == "__main__":  # pragma: no cover
    main()
