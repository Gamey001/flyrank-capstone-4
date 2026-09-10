"""The review workflow's logic, including its idempotency rule."""

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.db import models
from app.repositories.images import ImageRepository
from app.repositories.posts import PostRepository
from app.repositories.suggestions import ReviewRepository, SuggestionRepository


def record_review(
    session: Session,
    tenant_id: str,
    *,
    suggestion_id: str,
    decision: str,
    reviewer: str,
    note: Optional[str],
    idempotency_key: Optional[str],
) -> Tuple[models.Review, bool]:
    """Record a decision. Returns ``(review, created)``.

    A replayed key returns the original decision rather than recording a second
    one (shared requirement #5). Without a client-supplied key one is derived,
    so a double-clicked approve button still lands once.
    """
    suggestions = SuggestionRepository(session, tenant_id)
    suggestion = suggestions.get(suggestion_id)
    if suggestion is None:
        raise NotFoundError(f"suggestion {suggestion_id!r} not found")

    reviews = ReviewRepository(session, tenant_id)
    key = idempotency_key or f"{suggestion.id}:{reviewer}:{decision}"

    existing = reviews.get_by_idempotency_key(key)
    if existing is not None:
        if existing.suggestion_id != suggestion.id:
            raise ConflictError(
                "this Idempotency-Key was already used for a different "
                "suggestion; use a fresh key",
                details={"idempotency_key": key},
            )
        return existing, False

    review = reviews.add(
        models.Review(
            suggestion_id=suggestion.id,
            decision=decision,
            reviewer=reviewer,
            note=note,
            idempotency_key=key,
        )
    )
    session.flush()
    return review, True


def load_explanation(session: Session, tenant_id: str, suggestion_id: str):
    """Everything ``/explain`` needs, resolved in one place.

    Returns ``(suggestion, post, image, image_embedding, post_embedding,
    latest_review)``.
    """
    from app.repositories.embeddings import (
        OWNER_IMAGE,
        OWNER_POST,
        EmbeddingRepository,
    )

    suggestion = SuggestionRepository(session, tenant_id).get(suggestion_id)
    if suggestion is None:
        raise NotFoundError(f"suggestion {suggestion_id!r} not found")

    image = ImageRepository(session, tenant_id).get(suggestion.image_id)
    post = PostRepository(session, tenant_id).get(suggestion.post_id)
    if image is None or post is None:
        raise NotFoundError(
            f"suggestion {suggestion_id!r} references a missing post or image"
        )

    # The exact texts that were embedded are the most useful part of an
    # explanation: they show *what* the model compared, not just the score.
    embeddings = EmbeddingRepository(session, tenant_id).list_all()
    image_embedding = next(
        (
            row
            for row in embeddings
            if row.owner_type == OWNER_IMAGE and row.owner_id == image.id
        ),
        None,
    )
    post_embedding = next(
        (
            row
            for row in embeddings
            if row.owner_type == OWNER_POST and row.owner_id == post.id
        ),
        None,
    )
    review = ReviewRepository(session, tenant_id).latest_for_suggestion(
        suggestion.id
    )
    return suggestion, post, image, image_embedding, post_embedding, review


def list_review_queue(
    session: Session,
    tenant_id: str,
    *,
    verdict: Optional[str] = None,
    limit: int = 100,
) -> List[Tuple[models.Suggestion, models.Image]]:
    """Suggestions paired with their image, ordered post-then-rank."""
    suggestions = SuggestionRepository(session, tenant_id)
    images = ImageRepository(session, tenant_id)

    rows = suggestions.list_all()
    if verdict:
        rows = [row for row in rows if row.verdict == verdict]
    rows.sort(key=lambda row: (row.post_id, row.rank))

    out = []
    for row in rows[:limit]:
        image = images.get(row.image_id)
        if image is not None:
            out.append((row, image))
    return out


def list_reviews(
    session: Session,
    tenant_id: str,
    *,
    suggestion_id: Optional[str] = None,
    limit: int = 100,
) -> List[models.Review]:
    repo = ReviewRepository(session, tenant_id)
    rows = (
        repo.list_for_suggestion(suggestion_id)
        if suggestion_id
        else sorted(repo.list_all(), key=lambda row: row.created_at, reverse=True)
    )
    return rows[:limit]
