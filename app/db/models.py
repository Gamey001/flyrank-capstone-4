"""Persistence schema.

Layout follows the pipeline: corpus (tenants, images, image_tags, posts) →
AI artefacts (embeddings, ai_calls) → decisions (suggestions, reviews) →
orchestration (jobs).

Indexes are chosen from the queries the API actually issues; each one is
annotated with the access path it serves.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantScopedMixin, TimestampMixin, new_id

# --- status vocabularies (kept as plain strings so migrations stay portable) --
IMAGE_PENDING = "pending"
IMAGE_PROCESSING = "processing"
IMAGE_TAGGED = "tagged"
IMAGE_FAILED = "failed"

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECTED = "rejected"

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"


class Tenant(Base, TimestampMixin):
    """Isolation boundary. Every other table is scoped by ``tenant_id``."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Image(Base, TenantScopedMixin, TimestampMixin):
    """One corpus image plus the validated metadata the vision model produced."""

    __tablename__ = "images"
    __table_args__ = (
        # Ingestion is idempotent on content hash within a tenant.
        UniqueConstraint("tenant_id", "sha256", name="uq_images_tenant_sha256"),
        UniqueConstraint("tenant_id", "filename", name="uq_images_tenant_filename"),
        # "give me everything still to tag" — the batch job's claim query.
        Index("ix_images_tenant_status", "tenant_id", "status"),
        # review queue + category browsing in the admin table
        Index("ix_images_tenant_review", "tenant_id", "needs_review"),
        Index("ix_images_tenant_category", "tenant_id", "category"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=IMAGE_PENDING
    )

    # --- validated vision output (null until the batch job succeeds) ---
    subject: Mapped[Optional[str]] = mapped_column(String(200))
    category: Mapped[Optional[str]] = mapped_column(String(64))
    caption: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    attributes: Mapped[Optional[list]] = mapped_column(JSON)

    #: True when the model was unsure — flagged for a human, never silently used.
    needs_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    review_reason: Mapped[Optional[str]] = mapped_column(String(300))

    #: Raw provider payload, kept for auditing a bad classification.
    raw_response: Mapped[Optional[dict]] = mapped_column(JSON)
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    vision_model: Mapped[Optional[str]] = mapped_column(String(120))
    tagged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    tags: Mapped[list["ImageTag"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", lazy="selectin"
    )


class ImageTag(Base, TenantScopedMixin, TimestampMixin):
    """Normalised tags, so the guard can do set logic without parsing JSON."""

    __tablename__ = "image_tags"
    __table_args__ = (
        UniqueConstraint("image_id", "kind", "value", name="uq_image_tag"),
        # "which images carry this tag" — used by tag-filtered candidate lookup.
        Index("ix_image_tags_tenant_value", "tenant_id", "value"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    image_id: Mapped[str] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: one of: subject | category | attribute
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[str] = mapped_column(String(120), nullable=False)

    image: Mapped[Image] = relationship(back_populates="tags")


class Post(Base, TenantScopedMixin, TimestampMixin):
    """A blog post that needs an illustration."""

    __tablename__ = "posts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_posts_tenant_slug"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)


class Embedding(Base, TenantScopedMixin, TimestampMixin):
    """A vector in the shared semantic space.

    Stored as a JSON array: at ~50 images an in-DB array beats standing up
    pgvector, and it keeps SQLite (tests, keyless demo) and Postgres identical.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        # one vector per (owner, model) — re-embedding is an upsert, not a dup
        UniqueConstraint(
            "tenant_id",
            "owner_type",
            "owner_id",
            "model",
            name="uq_embeddings_owner_model",
        ),
        # the ranking query: load every image vector for a tenant+model
        Index(
            "ix_embeddings_tenant_owner_model", "tenant_id", "owner_type", "model"
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    #: image | post
    owner_type: Mapped[str] = mapped_column(String(20), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list] = mapped_column(JSON, nullable=False)
    #: text that was embedded — needed to explain a ranking after the fact
    source_text: Mapped[str] = mapped_column(Text, nullable=False, default="")


class Suggestion(Base, TenantScopedMixin, TimestampMixin):
    """One ranked (post, image) pairing together with the guard's verdict."""

    __tablename__ = "suggestions"
    __table_args__ = (
        # Re-running matching for a post overwrites rather than duplicates.
        UniqueConstraint(
            "tenant_id", "post_id", "image_id", name="uq_suggestions_post_image"
        ),
        # the read path: ranked list for a post
        Index("ix_suggestions_tenant_post_rank", "tenant_id", "post_id", "rank"),
        Index("ix_suggestions_tenant_verdict", "tenant_id", "verdict"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    post_id: Mapped[str] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[str] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    #: accepted | rejected — the mismatch guard's decision
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    #: [{code, message}] — always populated for a rejection
    reasons: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    #: full guard input/output, so /explain never has to recompute
    explanation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    post: Mapped[Post] = relationship()
    image: Mapped[Image] = relationship()


class Review(Base, TenantScopedMixin, TimestampMixin):
    """A human's approve/reject on a suggestion."""

    __tablename__ = "reviews"
    __table_args__ = (
        # Idempotency: a retried approve lands once (shared requirement #5).
        UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_reviews_idempotency"
        ),
        Index("ix_reviews_tenant_suggestion", "tenant_id", "suggestion_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    suggestion_id: Mapped[str] = mapped_column(
        ForeignKey("suggestions.id", ondelete="CASCADE"), nullable=False
    )
    #: approved | rejected
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(120), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(120), nullable=False)


class Job(Base, TenantScopedMixin, TimestampMixin):
    """A unit of slow/bulk work executed off the request path."""

    __tablename__ = "jobs"
    __table_args__ = (
        # Idempotency: submitting the same job twice returns the first one.
        UniqueConstraint("tenant_id", "dedupe_key", name="uq_jobs_dedupe"),
        # the worker's claim query
        Index("ix_jobs_status_scheduled", "status", "scheduled_at"),
        Index("ix_jobs_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=JOB_QUEUED
    )
    dedupe_key: Mapped[str] = mapped_column(String(200), nullable=False)

    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result: Mapped[Optional[dict]] = mapped_column(JSON)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # progress, so GET /jobs/{id} is useful while it runs
    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    last_error: Mapped[Optional[str]] = mapped_column(Text)


class AiCall(Base, TenantScopedMixin, TimestampMixin):
    """One row per AI request — the cost ledger (shared requirement #7)."""

    __tablename__ = "ai_calls"
    __table_args__ = (
        # "what did we spend today" — the budget guard's query
        Index("ix_ai_calls_tenant_created", "tenant_id", "created_at"),
        Index("ix_ai_calls_job", "job_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    #: vision | embedding
    operation: Mapped[str] = mapped_column(String(30), nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)

    #: what the call was for, e.g. "image:ab12" / "post:cd34"
    subject_ref: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    job_id: Mapped[Optional[str]] = mapped_column(String(32))

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: ok | error
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ok")
    error: Mapped[Optional[str]] = mapped_column(Text)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
