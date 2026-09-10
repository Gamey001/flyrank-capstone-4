"""Cost tracking and the budget guard (shared requirement #7)."""

import pytest

from app.core.errors import BudgetExceededError
from app.repositories.costs import AiCallRepository
from app.services.costs import OP_EMBEDDING, OP_VISION, CostTracker


def test_every_ai_call_is_attributed(session, seeded):
    calls = AiCallRepository(session, seeded).list_recent(limit=1000)
    assert calls, "the pipeline must leave a cost trail"

    for call in calls:
        assert call.operation in {OP_VISION, OP_EMBEDDING}
        assert call.provider and call.model
        # every call names what it was for and which job made it
        assert call.subject_ref.startswith(("image:", "post:"))
        assert call.job_id
        assert call.cost_usd >= 0.0


def test_vision_and_embedding_calls_cover_the_whole_corpus(session, seeded):
    from app.repositories.images import ImageRepository

    repo = AiCallRepository(session, seeded)
    image_count = ImageRepository(session, seeded).count()

    vision = [c for c in repo.list_recent(limit=1000) if c.operation == OP_VISION]
    assert len(vision) == image_count

    summary = repo.summary()
    assert summary["total_calls"] > image_count  # vision + embeddings
    assert summary["total_cost_usd"] > 0.0


def test_budget_guard_refuses_a_call_that_would_overspend(
    session, tenant_id, migrated
):
    tight = migrated.model_copy(update={"ai_daily_budget_usd": 0.0005})
    tracker = CostTracker(session, tenant_id, tight)

    # first call fits inside the budget
    tracker.assert_within_budget(tracker.price(OP_VISION))
    tracker.record(
        operation=OP_VISION,
        provider="fixture",
        model="m",
        subject_ref="image:1",
        input_tokens=10,
    )
    tracker.record(
        operation=OP_VISION,
        provider="fixture",
        model="m",
        subject_ref="image:2",
        input_tokens=10,
    )
    session.flush()

    with pytest.raises(BudgetExceededError) as exc:
        tracker.assert_within_budget(tracker.price(OP_VISION))
    assert "budget" in str(exc.value).lower()
    assert exc.value.status_code == 429


def test_zero_budget_disables_the_guard(session, tenant_id, migrated):
    unlimited = migrated.model_copy(update={"ai_daily_budget_usd": 0.0})
    CostTracker(session, tenant_id, unlimited).assert_within_budget(999.0)


def test_failed_calls_are_logged_at_zero_cost_but_stay_visible(
    session, tenant_id, migrated
):
    tracker = CostTracker(session, tenant_id, migrated)
    tracker.record(
        operation=OP_VISION,
        provider="gemini",
        model="gemini-2.0-flash",
        subject_ref="image:9",
        status="error",
        error="HTTP 503",
    )
    session.flush()

    summary = tracker.summary()
    assert summary["failed_calls"] == 1
    assert summary["total_cost_usd"] == 0.0


def test_budget_stops_a_batch_job_rather_than_failing_every_image(
    session, tenant_id, corpus_dir, migrated
):
    from app.repositories.jobs import claim_next_job
    from app.services.ingestion import ingest_directory
    from app.services.jobs import enqueue_job, run_job

    ingest_directory(session, tenant_id, corpus_dir)
    broke = migrated.model_copy(update={"ai_daily_budget_usd": 0.0004})
    enqueue_job(session, tenant_id, broke, kind="vision_tagging", dedupe_key="broke")
    session.commit()

    job = claim_next_job(session)
    run_job(session, job, broke)

    # over budget is a deliberate stop, not a retryable blip
    assert job.status == "failed"
    assert "budget" in (job.last_error or "").lower()
