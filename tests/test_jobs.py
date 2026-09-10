"""Background jobs: retries, resumability, alerting, idempotency."""

import pytest

from app.core.errors import ProviderError
from app.db import models
from app.repositories.images import ImageRepository
from app.repositories.jobs import JobRepository, claim_next_job
from app.services import jobs as jobs_service
from app.worker import drain, run_once


def test_vision_job_tags_the_whole_corpus_and_flags_low_confidence(
    session, seeded
):
    """PROBE 1."""
    images = ImageRepository(session, seeded)
    counts = images.counts_by_status()

    assert counts.get(models.IMAGE_TAGGED, 0) == images.count()
    assert counts.get(models.IMAGE_FAILED, 0) == 0

    flagged = images.list_filtered(needs_review=True, limit=500)
    assert flagged, "the corpus must contain at least one low-confidence image"
    for image in flagged:
        assert image.confidence < 0.70
        assert image.review_reason  # a human-readable explanation, not a bare flag

    # everything else was accepted with real, validated metadata
    for image in images.list_tagged():
        assert image.subject and image.category and image.caption
        assert 0.0 <= image.confidence <= 1.0
        assert image.vision_model


def test_enqueue_is_idempotent_on_dedupe_key(session, tenant_id, migrated):
    first, created_a = jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="nightly"
    )
    second, created_b = jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="nightly"
    )
    assert created_a is True and created_b is False
    assert first.id == second.id
    assert JobRepository(session, tenant_id).count() == 1


def test_unknown_job_kind_is_rejected(session, tenant_id, migrated):
    from app.core.errors import AppError

    with pytest.raises(AppError):
        jobs_service.enqueue_job(
            session, tenant_id, migrated, kind="make_coffee", dedupe_key="k"
        )


def test_retryable_failure_requeues_with_backoff(
    session, tenant_id, migrated, monkeypatch
):
    calls = {"n": 0}

    def flaky(_session, _job, _settings):
        calls["n"] += 1
        raise ProviderError("upstream hiccup", retryable=True)

    monkeypatch.setitem(jobs_service.HANDLERS, "vision_tagging", flaky)
    job, _ = jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="flaky"
    )
    session.commit()

    claimed = claim_next_job(session)
    jobs_service.run_job(session, claimed, migrated)

    assert calls["n"] == 1
    assert claimed.status == models.JOB_QUEUED   # queued again, not failed
    assert claimed.attempts == 1
    assert "upstream hiccup" in claimed.last_error


def test_exhausted_retries_fail_the_job_and_raise_an_alert(
    session, tenant_id, migrated, monkeypatch
):
    alerts = []

    def always_fails(_session, _job, _settings):
        raise ProviderError("provider is down", retryable=True)

    monkeypatch.setitem(jobs_service.HANDLERS, "vision_tagging", always_fails)
    monkeypatch.setattr(
        jobs_service, "send_job_failure_alert", lambda **kw: alerts.append(kw)
    )

    job, _ = jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="doomed"
    )
    job.max_attempts = 2
    session.commit()

    for _ in range(job.max_attempts):
        # Fast-forward past the retry backoff so the test does not sleep.
        job.scheduled_at = job.created_at
        session.flush()

        claimed = claim_next_job(session)
        assert claimed is not None
        jobs_service.run_job(session, claimed, migrated)
        session.commit()

    session.refresh(job)
    assert job.status == models.JOB_FAILED
    assert job.attempts == 2
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "vision_tagging"
    assert "provider is down" in alerts[0]["error"]


def test_non_retryable_failure_fails_immediately(
    session, tenant_id, migrated, monkeypatch
):
    def hard_fail(_session, _job, _settings):
        raise ProviderError("image file is corrupt", retryable=False)

    monkeypatch.setitem(jobs_service.HANDLERS, "vision_tagging", hard_fail)
    monkeypatch.setattr(jobs_service, "send_job_failure_alert", lambda **kw: None)

    jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="corrupt"
    )
    session.commit()

    claimed = claim_next_job(session)
    jobs_service.run_job(session, claimed, migrated)
    assert claimed.status == models.JOB_FAILED
    assert claimed.attempts == 1  # no retry burned


def test_tagging_is_resumable_and_does_not_repay_for_done_work(
    session, seeded, migrated
):
    """A second run of the vision job costs nothing: nothing is pending."""
    from app.repositories.costs import AiCallRepository

    calls_before = AiCallRepository(session, seeded).summary()["total_calls"]

    jobs_service.enqueue_job(
        session, seeded, migrated, kind="vision_tagging", dedupe_key="second-pass"
    )
    session.commit()
    drain(migrated)
    session.expire_all()

    calls_after = AiCallRepository(session, seeded).summary()["total_calls"]
    assert calls_after == calls_before


def test_worker_returns_none_when_the_queue_is_empty(session, tenant_id, migrated):
    session.commit()
    assert run_once(migrated) is None


def test_stale_running_jobs_are_reclaimed(session, tenant_id, migrated):
    from datetime import timedelta

    from app.db.base import utcnow
    from app.worker import reclaim_stale_jobs

    job, _ = jobs_service.enqueue_job(
        session, tenant_id, migrated, kind="vision_tagging", dedupe_key="orphan"
    )
    job.status = models.JOB_RUNNING
    job.started_at = utcnow() - timedelta(hours=2)
    session.commit()

    assert reclaim_stale_jobs(migrated) == 1
    session.expire_all()
    session.refresh(job)
    assert job.status == models.JOB_QUEUED
