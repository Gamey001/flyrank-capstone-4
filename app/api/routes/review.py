"""The review workflow: approve, reject, and inspect why."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_tenant_id
from app.api.presenters import suggestion_row
from app.schemas.api import (
    ExplainResponse,
    ReasonOut,
    ReviewCreate,
    ReviewOut,
    SuggestionOut,
)
from app.services import review_service

router = APIRouter(tags=["review"])


@router.get(
    "/suggestions",
    response_model=List[SuggestionOut],
    summary="The review queue",
)
def list_suggestions(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    verdict: Optional[str] = Query(default=None, description="accepted | rejected"),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[SuggestionOut]:
    rows = review_service.list_review_queue(
        session, tenant_id, verdict=verdict, limit=limit
    )
    return [suggestion_row(suggestion, image) for suggestion, image in rows]


@router.get(
    "/suggestions/{suggestion_id}/explain",
    response_model=ExplainResponse,
    summary="Why was this image selected — or refused?",
)
def explain(
    suggestion_id: str,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> ExplainResponse:
    (
        suggestion,
        post,
        image,
        image_embedding,
        post_embedding,
        review,
    ) = review_service.load_explanation(session, tenant_id, suggestion_id)

    return ExplainResponse(
        suggestion_id=suggestion.id,
        post_id=post.id,
        post_title=post.title,
        image_id=image.id,
        filename=image.filename,
        rank=suggestion.rank,
        similarity=round(suggestion.similarity, 4),
        verdict=suggestion.verdict,
        reasons=[ReasonOut(**r) for r in (suggestion.reasons or [])],
        signals=suggestion.explanation or {},
        image_embedding_text=image_embedding.source_text if image_embedding else None,
        post_embedding_text=post_embedding.source_text if post_embedding else None,
        review=ReviewOut.model_validate(review) if review else None,
    )


@router.post(
    "/reviews",
    response_model=ReviewOut,
    status_code=status.HTTP_201_CREATED,
    summary="Approve or reject a suggested pairing (idempotent)",
)
def create_review(
    payload: ReviewCreate,
    response: Response,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
        description=(
            "Replaying the same key returns the original review with 200 "
            "instead of recording a second decision."
        ),
    ),
) -> ReviewOut:
    review, created = review_service.record_review(
        session,
        tenant_id,
        suggestion_id=payload.suggestion_id,
        decision=payload.decision,
        reviewer=payload.reviewer,
        note=payload.note,
        idempotency_key=idempotency_key,
    )
    if not created:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replay"] = "true"
    return ReviewOut.model_validate(review)


@router.get("/reviews", response_model=List[ReviewOut], summary="Review history")
def list_reviews(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    suggestion_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[ReviewOut]:
    rows = review_service.list_reviews(
        session, tenant_id, suggestion_id=suggestion_id, limit=limit
    )
    return [ReviewOut.model_validate(row) for row in rows]
