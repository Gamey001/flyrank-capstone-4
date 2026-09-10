"""The matching engine: rank candidates, then put every one through the guard.

Order matters. Ranking answers "which image is closest?"; the guard answers
"is the closest image actually right?". Running the guard *after* ranking —
on every candidate, not just the winner — is what lets the API return an
explained refusal instead of an empty list.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.db import models
from app.repositories.embeddings import OWNER_IMAGE, EmbeddingRepository
from app.repositories.images import ImageRepository
from app.repositories.suggestions import SuggestionRepository
from app.services import guard as guard_module
from app.services.embedding_service import EmbeddingService
from app.services.similarity import rank_by_similarity
from app.core.logging import safe_extra

logger = logging.getLogger("app.matching")


@dataclass
class ScoredCandidate:
    image: models.Image
    similarity: float
    verdict: guard_module.GuardVerdict
    rank: int


@dataclass
class MatchResult:
    post: models.Post
    intent: guard_module.PostIntent
    candidates: List[ScoredCandidate]
    embedding_model: str

    @property
    def accepted(self) -> List[ScoredCandidate]:
        return [c for c in self.candidates if c.verdict.accepted]

    @property
    def has_match(self) -> bool:
        return bool(self.accepted)


class MatchingService:
    def __init__(
        self,
        session: Session,
        tenant_id: str,
        settings: Settings,
        embedding_service: EmbeddingService,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self.settings = settings
        self.embeddings = embedding_service
        self.embedding_repo = EmbeddingRepository(session, tenant_id)
        self.images = ImageRepository(session, tenant_id)
        self.suggestions = SuggestionRepository(session, tenant_id)
        self.thresholds = thresholds_from_settings(settings)

    # --- core ------------------------------------------------------------
    def match_post(self, post: models.Post, limit: int = 10) -> MatchResult:
        intent = guard_module.PostIntent.from_text(post.id, post.title, post.body)
        model_name = self.embeddings.model_name

        post_vector = self.embeddings.ensure_post_embedding(post).vector
        image_vectors = self.embedding_repo.vectors_by_owner(OWNER_IMAGE, model_name)

        if not image_vectors:
            logger.warning(
                "no_image_embeddings",
                extra=safe_extra({"tenant_id": self.tenant_id, "model": model_name}),
            )
            return MatchResult(post, intent, [], model_name)

        images_by_id = {img.id: img for img in self.images.list_tagged()}
        # Only rank images we still hold a tagged row for; a stale vector from a
        # deleted image must never surface.
        ranked = [
            (image_id, score)
            for image_id, score in rank_by_similarity(
                post_vector,
                image_vectors,
                # Filenames, not row ids: ties must resolve the same way on
                # every machine and every re-seed.
                tiebreak={img_id: img.filename for img_id, img in images_by_id.items()},
            )
            if image_id in images_by_id
        ][:limit]

        candidates: List[ScoredCandidate] = []
        for position, (image_id, score) in enumerate(ranked, start=1):
            image = images_by_id[image_id]
            runner_up = ranked[position][1] if position < len(ranked) else None
            verdict = guard_module.evaluate(
                intent,
                to_candidate(image),
                score,
                self.thresholds,
                runner_up_similarity=runner_up if position == 1 else None,
            )
            candidates.append(ScoredCandidate(image, score, verdict, position))

        return MatchResult(post, intent, candidates, model_name)

    def match_and_persist(self, post: models.Post, limit: int = 10) -> MatchResult:
        """Rank, guard, and store the outcome so the review API has rows to act on."""
        result = self.match_post(post, limit=limit)
        for candidate in result.candidates:
            self.suggestions.upsert(
                post_id=post.id,
                image_id=candidate.image.id,
                rank=candidate.rank,
                similarity=candidate.similarity,
                verdict=(
                    models.VERDICT_ACCEPTED
                    if candidate.verdict.accepted
                    else models.VERDICT_REJECTED
                ),
                reasons=[r.as_dict() for r in candidate.verdict.reasons],
                explanation=candidate.verdict.signals,
            )
        self.session.flush()
        logger.info(
            "post_matched",
            extra=safe_extra({
                "post_id": post.id,
                "candidates": len(result.candidates),
                "accepted": len(result.accepted),
            }),
        )
        return result

    def check_pair(self, post: models.Post, image: models.Image) -> ScoredCandidate:
        """Force one specific image through the guard for a post (PROBE 3).

        Bypasses ranking entirely: the point is to prove that a candidate the
        ranker likes can still be refused on tag grounds.
        """
        model_name = self.embeddings.model_name
        post_vector = self.embeddings.ensure_post_embedding(post).vector
        image_embedding = self.embedding_repo.get_for(
            OWNER_IMAGE, image.id, model_name
        )
        if image_embedding is None:
            raise NotFoundError(
                f"image {image.id} has no embedding for model {model_name}; run "
                "the embed_images job first",
                details={"image_id": image.id, "model": model_name},
            )
        from app.services.similarity import cosine_similarity

        score = cosine_similarity(post_vector, image_embedding.vector)
        intent = guard_module.PostIntent.from_text(post.id, post.title, post.body)
        verdict = guard_module.evaluate(
            intent, to_candidate(image), score, self.thresholds
        )
        return ScoredCandidate(image, score, verdict, rank=0)

    def no_match_payload(self, result: MatchResult) -> Dict[str, Any]:
        return guard_module.no_match_explanation(
            result.intent,
            [c.verdict for c in result.candidates],
            self.thresholds,
        )


def thresholds_from_settings(settings: Settings) -> guard_module.GuardThresholds:
    return guard_module.GuardThresholds(
        similarity=settings.similarity_threshold,
        min_confidence=settings.min_vision_confidence,
        low_confidence_flag=settings.low_confidence_threshold,
        ambiguity_margin=settings.ambiguity_margin,
    )


def to_candidate(image: models.Image) -> guard_module.Candidate:
    return guard_module.Candidate(
        image_id=image.id,
        subject=image.subject,
        category=image.category,
        caption=image.caption or "",
        confidence=image.confidence,
        attributes=tuple(image.attributes or ()),
        needs_review=bool(image.needs_review),
        is_tagged=image.status == models.IMAGE_TAGGED,
    )
