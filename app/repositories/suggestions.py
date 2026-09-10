"""Suggestion + review persistence."""

from typing import List, Optional

from app.db import models
from app.repositories.base import TenantRepository


class SuggestionRepository(TenantRepository[models.Suggestion]):
    model = models.Suggestion

    def list_for_post(self, post_id: str, limit: int = 20) -> List[models.Suggestion]:
        return list(
            self.session.scalars(
                self._scoped()
                .where(models.Suggestion.post_id == post_id)
                .order_by(models.Suggestion.rank)
                .limit(limit)
            )
        )

    def get_pair(self, post_id: str, image_id: str) -> Optional[models.Suggestion]:
        return self.session.scalars(
            self._scoped().where(
                models.Suggestion.post_id == post_id,
                models.Suggestion.image_id == image_id,
            )
        ).first()

    def upsert(
        self,
        *,
        post_id: str,
        image_id: str,
        rank: int,
        similarity: float,
        verdict: str,
        reasons: list,
        explanation: dict,
    ) -> models.Suggestion:
        """Re-running matching for a post updates rows instead of duplicating."""
        existing = self.get_pair(post_id, image_id)
        if existing is None:
            existing = self.add(
                models.Suggestion(
                    post_id=post_id,
                    image_id=image_id,
                    rank=rank,
                    similarity=similarity,
                    verdict=verdict,
                    reasons=reasons,
                    explanation=explanation,
                )
            )
            self.session.flush()
            return existing
        existing.rank = rank
        existing.similarity = similarity
        existing.verdict = verdict
        existing.reasons = reasons
        existing.explanation = explanation
        return existing

    def clear_for_post(self, post_id: str) -> None:
        self.session.query(models.Suggestion).filter(
            models.Suggestion.tenant_id == self.tenant_id,
            models.Suggestion.post_id == post_id,
        ).delete(synchronize_session=False)


class ReviewRepository(TenantRepository[models.Review]):
    model = models.Review

    def get_by_idempotency_key(self, key: str) -> Optional[models.Review]:
        return self.session.scalars(
            self._scoped().where(models.Review.idempotency_key == key)
        ).first()

    def list_for_suggestion(self, suggestion_id: str) -> List[models.Review]:
        return list(
            self.session.scalars(
                self._scoped()
                .where(models.Review.suggestion_id == suggestion_id)
                .order_by(models.Review.created_at.desc())
            )
        )

    def latest_for_suggestion(self, suggestion_id: str) -> Optional[models.Review]:
        rows = self.list_for_suggestion(suggestion_id)
        return rows[0] if rows else None
