"""ORM/domain objects → API models.

Kept out of the routes so the HTTP layer stays a thin translation of services.
"""

from typing import Optional

from app.db import models
from app.schemas import api as schemas
from app.services.matching import MatchResult, ScoredCandidate


def suggestion_from_candidate(
    candidate: ScoredCandidate, suggestion_id: Optional[str] = None
) -> schemas.SuggestionOut:
    image = candidate.image
    return schemas.SuggestionOut(
        suggestion_id=suggestion_id,
        image_id=image.id,
        filename=image.filename,
        rank=candidate.rank,
        similarity=round(candidate.similarity, 4),
        accepted=candidate.verdict.accepted,
        subject=image.subject,
        category=image.category,
        caption=image.caption,
        confidence=image.confidence,
        needs_review=bool(image.needs_review),
        reasons=[
            schemas.ReasonOut(**reason.as_dict())
            for reason in candidate.verdict.reasons
        ],
    )


def match_response(
    result: MatchResult,
    *,
    no_match: Optional[dict] = None,
    suggestion_ids: Optional[dict] = None,
) -> schemas.MatchResponse:
    suggestion_ids = suggestion_ids or {}
    accepted = [
        suggestion_from_candidate(c, suggestion_ids.get(c.image.id))
        for c in result.candidates
        if c.verdict.accepted
    ]
    rejected = [
        suggestion_from_candidate(c, suggestion_ids.get(c.image.id))
        for c in result.candidates
        if not c.verdict.accepted
    ]
    return schemas.MatchResponse(
        post_id=result.post.id,
        post_slug=result.post.slug,
        post_title=result.post.title,
        embedding_model=result.embedding_model,
        has_confident_match=bool(accepted),
        best_match=accepted[0] if accepted else None,
        suggestions=accepted,
        rejected=rejected,
        no_match=schemas.NoMatchOut(**no_match) if no_match else None,
    )


def suggestion_row(
    suggestion: models.Suggestion, image: models.Image
) -> schemas.SuggestionOut:
    return schemas.SuggestionOut(
        suggestion_id=suggestion.id,
        image_id=image.id,
        filename=image.filename,
        rank=suggestion.rank,
        similarity=round(suggestion.similarity, 4),
        accepted=suggestion.verdict == models.VERDICT_ACCEPTED,
        subject=image.subject,
        category=image.category,
        caption=image.caption,
        confidence=image.confidence,
        needs_review=bool(image.needs_review),
        reasons=[schemas.ReasonOut(**r) for r in (suggestion.reasons or [])],
    )
