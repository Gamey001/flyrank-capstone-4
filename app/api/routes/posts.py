"""Post endpoints, including the ranked-image read path."""

from typing import List

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_matching_service, get_tenant_id
from app.api.presenters import match_response
from app.core.errors import NotFoundError
from app.repositories.images import ImageRepository
from app.schemas.api import (
    GuardCheckResponse,
    MatchResponse,
    PostCreate,
    PostOut,
    ReasonOut,
)
from app.services import content
from app.services.matching import MatchingService

router = APIRouter(prefix="/posts", tags=["posts"])


@router.post(
    "",
    response_model=PostOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a post",
)
def create_post(
    payload: PostCreate,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> PostOut:
    post = content.create_post(session, tenant_id, **payload.model_dump())
    return PostOut.model_validate(post)


@router.get("", response_model=List[PostOut], summary="List posts")
def list_posts(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[PostOut]:
    rows = content.list_posts(session, tenant_id, limit=limit, offset=offset)
    return [PostOut.model_validate(row) for row in rows]


@router.get("/{post_ref}", response_model=PostOut, summary="Get a post by id or slug")
def get_post(
    post_ref: str,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
) -> PostOut:
    return PostOut.model_validate(content.resolve_post(session, tenant_id, post_ref))


@router.get(
    "/{post_ref}/images",
    response_model=MatchResponse,
    summary="Ranked, guarded image suggestions for a post",
    response_description=(
        "Accepted suggestions best-first. When the guard refuses every "
        "candidate, `has_confident_match` is false and `no_match` explains why."
    ),
)
def suggest_images(
    post_ref: str,
    response: Response,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    matching: MatchingService = Depends(get_matching_service),
    # The cap is generous on purpose: at corpus scale, asking for the full
    # ranking (to see where the wolf landed) is a normal thing to want.
    limit: int = Query(default=10, ge=1, le=200),
    persist: bool = Query(
        default=True,
        description="Store the outcome as reviewable suggestion rows.",
    ),
) -> MatchResponse:
    post = content.resolve_post(session, tenant_id, post_ref)
    result = (
        matching.match_and_persist(post, limit=limit)
        if persist
        else matching.match_post(post, limit=limit)
    )

    suggestion_ids = {}
    if persist:
        suggestion_ids = {
            row.image_id: row.id
            for row in matching.suggestions.list_for_post(post.id, limit=limit)
        }

    no_match = None
    if not result.has_match:
        no_match = matching.no_match_payload(result)
        # A refusal is a successful, deliberate answer — 200 with an
        # explanation, not an error.
        response.headers["X-Match-Verdict"] = "no-confident-match"

    return match_response(result, no_match=no_match, suggestion_ids=suggestion_ids)


@router.post(
    "/{post_ref}/images/{image_id}/check",
    response_model=GuardCheckResponse,
    summary="Force one image through the mismatch guard",
    response_description=(
        "Runs the guard on a specific pairing regardless of its rank — the "
        "wolf-on-a-fox-post probe."
    ),
)
def check_pair(
    post_ref: str,
    image_id: str,
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    matching: MatchingService = Depends(get_matching_service),
) -> GuardCheckResponse:
    post = content.resolve_post(session, tenant_id, post_ref)
    image = ImageRepository(session, tenant_id).get(image_id)
    if image is None:
        raise NotFoundError(f"image {image_id!r} not found")

    candidate = matching.check_pair(post, image)
    return GuardCheckResponse(
        post_id=post.id,
        post_title=post.title,
        image_id=image.id,
        filename=image.filename,
        similarity=round(candidate.similarity, 4),
        accepted=candidate.verdict.accepted,
        reasons=[ReasonOut(**r.as_dict()) for r in candidate.verdict.reasons],
        signals=candidate.verdict.signals,
    )
