"""Seed the demo tenant and run the whole pipeline.

    python -m scripts.seed

Steps: migrate → create the tenant → ingest the corpus → load the posts →
queue ``full_pipeline`` → drain the worker → print the state a reviewer wants
to see. Safe to run repeatedly: every step is idempotent.
"""

import argparse
import json
import os
import sys
from typing import List

from alembic import command
from alembic.config import Config

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db import models
from app.db.session import session_scope
from app.repositories.images import ImageRepository
from app.repositories.posts import PostRepository
from app.repositories.tenants import ensure_tenant
from app.services.costs import CostTracker
from app.services.ingestion import ingest_directory
from app.services.jobs import KIND_FULL_PIPELINE, enqueue_job
from app.worker import drain

DEFAULT_POSTS = "data/seed/posts.json"


def migrate() -> None:
    config = Config("alembic.ini")
    command.upgrade(config, "head")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", default=DEFAULT_POSTS)
    parser.add_argument("--tenant", default=None)
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Ingest and load posts but do not run the AI jobs.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    tenant_id = args.tenant or settings.default_tenant

    print("1/5  applying migrations")
    migrate()

    print(f"2/5  ensuring tenant {tenant_id!r}")
    with session_scope() as session:
        ensure_tenant(session, tenant_id, name="Demo workspace")

    print(f"3/5  ingesting corpus from {settings.images_dir}")
    if not os.path.isdir(settings.images_dir) or not os.listdir(settings.images_dir):
        print(
            f"     ! {settings.images_dir} is empty. Run either\n"
            "         python -m scripts.fetch_corpus            (real photos)\n"
            "         python -m scripts.make_placeholder_corpus (offline tiles)"
        )
        return 1
    with session_scope() as session:
        summary = ingest_directory(session, tenant_id, settings.images_dir)
    print(f"     images: {summary}")

    print(f"4/5  loading posts from {args.posts}")
    with open(args.posts, "r", encoding="utf-8") as fh:
        posts = json.load(fh)["posts"]
    with session_scope() as session:
        repo = PostRepository(session, tenant_id)
        created = 0
        for entry in posts:
            if repo.get_by_slug(entry["slug"]) is None:
                repo.add(models.Post(**entry))
                created += 1
    print(f"     posts: {created} created, {len(posts) - created} already present")

    if args.skip_pipeline:
        print("5/5  skipped (--skip-pipeline)")
        return 0

    print("5/5  running the full pipeline (vision → embeddings → matching)")
    with session_scope() as session:
        job, _ = enqueue_job(
            session,
            tenant_id,
            settings,
            kind=KIND_FULL_PIPELINE,
            dedupe_key="seed:full_pipeline",
        )
        job_id = job.id
    ran = drain(settings)
    print(f"     worker ran {ran} job(s)")

    _report(tenant_id, job_id)
    return 0


def _report(tenant_id: str, job_id: str) -> None:
    from app.repositories.jobs import JobRepository

    settings = get_settings()
    with session_scope() as session:
        job = JobRepository(session, tenant_id).get(job_id)
        images = ImageRepository(session, tenant_id)
        costs = CostTracker(session, tenant_id, settings).summary()
        by_status = images.counts_by_status()
        flagged = images.list_filtered(needs_review=True, limit=500)

        print("\n--- seed complete -------------------------------------------")
        print(f"job {job_id} → {job.status} (attempts {job.attempts})")
        print(f"images by status: {by_status}")
        print(f"flagged for review: {len(flagged)}")
        for image in flagged:
            print(
                f"  · {image.filename}: confidence {image.confidence:.2f} — "
                f"{image.review_reason}"
            )
        print(
            f"AI calls: {costs['total_calls']} · "
            f"cost ${costs['total_cost_usd']:.6f} of "
            f"${costs['daily_budget_usd']:.2f} daily budget"
        )
        print("\nTry:")
        print("  curl -s localhost:8000/v1/posts/red-foxes-in-deep-snow/images | jq")
        print("  python -m scripts.run_eval")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
