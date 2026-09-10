"""Test fixtures.

Every test runs against a throwaway SQLite database created from the Alembic
migrations — not ``create_all`` — so a migration that drifts from the models
fails the suite rather than production.
"""

import os
from typing import Iterator

import pytest

# Configure before anything imports app.core.config: settings are cached.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("VISION_PROVIDER", "stub")
os.environ.setdefault("EMBEDDING_PROVIDER", "stub")
os.environ.setdefault("WORKER_ENABLED", "false")
os.environ.setdefault("CORPUS_MANIFEST", "data/corpus/manifest.json")


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    from app.core.config import get_settings

    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_path}")
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("WORKER_ENABLED", "false")
    get_settings.cache_clear()

    from app.db.session import reset_engine

    reset_engine()
    yield get_settings()

    get_settings.cache_clear()
    reset_engine()


@pytest.fixture()
def migrated(settings):
    """Apply migrations to the throwaway database."""
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    command.upgrade(config, "head")
    return settings


@pytest.fixture()
def session(migrated) -> Iterator:
    from app.db.session import get_session_factory

    db = get_session_factory()()
    try:
        yield db
        db.commit()
    finally:
        db.close()


@pytest.fixture()
def tenant_id(session, migrated) -> str:
    from app.repositories.tenants import ensure_tenant

    ensure_tenant(session, migrated.default_tenant, name="Test tenant")
    session.commit()
    return migrated.default_tenant


@pytest.fixture()
def corpus_dir(migrated) -> str:
    """A small on-disk corpus generated from the manifest."""
    from scripts.make_placeholder_corpus import main as make_corpus

    out = migrated.images_dir
    os.makedirs(out, exist_ok=True)
    make_corpus(["--out", out, "--size", "16"])
    return out


@pytest.fixture()
def seeded(session, tenant_id, corpus_dir, migrated):
    """Corpus ingested, posts loaded, whole pipeline run."""
    import json

    from app.db import models
    from app.repositories.posts import PostRepository
    from app.services.ingestion import ingest_directory
    from app.services.jobs import KIND_FULL_PIPELINE, enqueue_job
    from app.worker import drain

    ingest_directory(session, tenant_id, corpus_dir)
    with open("data/seed/posts.json", "r", encoding="utf-8") as fh:
        posts = json.load(fh)["posts"]
    repo = PostRepository(session, tenant_id)
    for entry in posts:
        repo.add(models.Post(**entry))
    enqueue_job(
        session,
        tenant_id,
        migrated,
        kind=KIND_FULL_PIPELINE,
        dedupe_key="test:pipeline",
    )
    session.commit()
    drain(migrated)
    session.expire_all()
    return tenant_id


@pytest.fixture()
def client(migrated):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(migrated)) as test_client:
        yield test_client


@pytest.fixture()
def seeded_client(client, seeded):
    return client
