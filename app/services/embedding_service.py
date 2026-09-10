"""Embedding generation for both sides of the match."""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ProviderError
from app.db import models
from app.providers.base import EmbeddingProvider
from app.repositories.embeddings import OWNER_IMAGE, OWNER_POST, EmbeddingRepository
from app.services.costs import OP_EMBEDDING, CostTracker

logger = logging.getLogger("app.embeddings")


def image_embedding_text(image: models.Image) -> str:
    """What represents an image in the shared space: its *meaning*, not its name.

    Filenames are deliberately excluded — matching on ``red-fox-snow-01.jpg``
    would be keyword search wearing a vector costume.
    """
    parts = [image.caption or "", image.subject or "", image.category or ""]
    parts.extend(image.attributes or [])
    return " ".join(p for p in parts if p).strip()


def post_embedding_text(post: models.Post) -> str:
    # Title twice: it carries the post's intent far more reliably than the body.
    return f"{post.title}. {post.title}. {post.body}".strip()


class EmbeddingService:
    def __init__(
        self,
        session: Session,
        tenant_id: str,
        settings: Settings,
        provider: EmbeddingProvider,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self.settings = settings
        self.provider = provider
        self.repo = EmbeddingRepository(session, tenant_id)
        self.costs = cost_tracker or CostTracker(session, tenant_id, settings)

    @property
    def model_name(self) -> str:
        return getattr(self.provider, "model", "unknown")

    def embed_text(
        self, text: str, *, subject_ref: str, job_id: Optional[str] = None
    ) -> List[float]:
        projected = self.costs.price(
            OP_EMBEDDING, input_tokens=max(1, len(text.split()))
        )
        self.costs.assert_within_budget(projected)
        try:
            result = self.provider.embed(text)
        except ProviderError as exc:
            self.costs.record(
                operation=OP_EMBEDDING,
                provider=getattr(self.provider, "name", "unknown"),
                model=self.model_name,
                subject_ref=subject_ref,
                job_id=job_id,
                status="error",
                error=str(exc),
            )
            raise
        self.costs.record(
            operation=OP_EMBEDDING,
            provider=getattr(self.provider, "name", "unknown"),
            model=result.model,
            subject_ref=subject_ref,
            job_id=job_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=result.usage.latency_ms,
        )
        return result.vector

    def embed_image(
        self, image: models.Image, *, job_id: Optional[str] = None
    ) -> models.Embedding:
        text = image_embedding_text(image)
        vector = self.embed_text(
            text, subject_ref=f"image:{image.id}", job_id=job_id
        )
        return self.repo.upsert(
            owner_type=OWNER_IMAGE,
            owner_id=image.id,
            embedding_model=self.model_name,
            vector=vector,
            source_text=text,
        )

    def embed_post(
        self, post: models.Post, *, job_id: Optional[str] = None
    ) -> models.Embedding:
        text = post_embedding_text(post)
        vector = self.embed_text(text, subject_ref=f"post:{post.id}", job_id=job_id)
        return self.repo.upsert(
            owner_type=OWNER_POST,
            owner_id=post.id,
            embedding_model=self.model_name,
            vector=vector,
            source_text=text,
        )

    def ensure_post_embedding(self, post: models.Post) -> models.Embedding:
        """Lazily embed a post on first read — a single fast call, inline is fine.

        Bulk embedding still belongs to the batch job; this only covers the
        "post created a second ago, now asking for images" path.
        """
        existing = self.repo.get_for(OWNER_POST, post.id, self.model_name)
        return existing if existing is not None else self.embed_post(post)
