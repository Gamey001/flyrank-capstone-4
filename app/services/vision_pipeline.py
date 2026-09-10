"""Turning images into validated, tagged metadata.

The contract this module enforces:

* the provider's answer is parsed through ``ImageTags`` before anything else
  touches it — an unparseable answer is an error, never a partial write;
* a low-confidence classification is *flagged*, not accepted;
* every call — success or failure — lands in the cost ledger.
"""

import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ProviderError
from app.db import models
from app.db.base import utcnow
from app.providers.base import VisionProvider
from app.repositories.images import ImageRepository
from app.services.costs import OP_VISION, CostTracker
from app.core.logging import safe_extra

logger = logging.getLogger("app.vision")


class VisionService:
    def __init__(
        self,
        session: Session,
        tenant_id: str,
        settings: Settings,
        provider: VisionProvider,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self.settings = settings
        self.provider = provider
        self.images = ImageRepository(session, tenant_id)
        self.costs = cost_tracker or CostTracker(session, tenant_id, settings)

    def tag_image(
        self, image: models.Image, *, job_id: Optional[str] = None
    ) -> models.Image:
        """Classify one image and persist the validated result.

        Raises ``ProviderError`` (retryable or not) on failure; the caller
        decides whether to retry. The image row is left in a truthful state
        either way.
        """
        self.costs.assert_within_budget(
            self.costs.price(OP_VISION)
        )

        image.status = models.IMAGE_PROCESSING
        self.session.flush()

        try:
            result = self.provider.describe_image(image.path)
        except ProviderError as exc:
            self.costs.record(
                operation=OP_VISION,
                provider=getattr(self.provider, "name", "unknown"),
                model=getattr(self.provider, "model", "unknown"),
                subject_ref=f"image:{image.id}",
                job_id=job_id,
                status="error",
                error=str(exc),
            )
            image.status = models.IMAGE_FAILED
            image.last_error = str(exc)[:2000]
            self.session.flush()
            raise

        self.costs.record(
            operation=OP_VISION,
            provider=getattr(self.provider, "name", "unknown"),
            model=result.model,
            subject_ref=f"image:{image.id}",
            job_id=job_id,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=result.usage.latency_ms,
            meta=result.usage.meta or None,
        )

        tags = result.tags
        needs_review, reason = self._review_decision(tags.confidence)

        image.subject = tags.subject
        image.category = tags.category
        image.caption = tags.caption
        image.confidence = tags.confidence
        image.attributes = list(tags.attributes)
        image.raw_response = result.raw
        image.vision_model = result.model
        image.needs_review = needs_review
        image.review_reason = reason
        image.status = models.IMAGE_TAGGED
        image.last_error = None
        image.tagged_at = utcnow()

        self.images.replace_tags(image, self._tag_rows(tags))
        self.session.flush()

        logger.info(
            "image_tagged",
            extra=safe_extra({
                "image_id": image.id,
                "filename": image.filename,
                "subject": tags.subject,
                "confidence": tags.confidence,
                "needs_review": needs_review,
                "job_id": job_id,
            }),
        )
        return image

    def _review_decision(self, confidence: float) -> Tuple[bool, Optional[str]]:
        """Below the flag threshold the classification is not trusted outright."""
        if confidence < self.settings.min_vision_confidence:
            return True, (
                f"Vision confidence {confidence:.2f} is below the "
                f"{self.settings.min_vision_confidence:.2f} usable minimum; "
                "this image is never auto-suggested."
            )
        if confidence < self.settings.low_confidence_threshold:
            return True, (
                f"Vision confidence {confidence:.2f} is below the "
                f"{self.settings.low_confidence_threshold:.2f} trust threshold; "
                "flagged for human confirmation."
            )
        return False, None

    @staticmethod
    def _tag_rows(tags) -> List[Tuple[str, str]]:
        rows = [("subject", tags.subject), ("category", tags.category)]
        rows.extend(("attribute", attr) for attr in tags.attributes)
        return rows
