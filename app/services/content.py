"""Post lifecycle — the logic the HTTP layer delegates to.

Small, but it keeps the routes free of ORM construction: a route parses,
delegates and presents, and nothing else.
"""

from typing import List

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.db import models
from app.repositories.posts import PostRepository


def create_post(
    session: Session, tenant_id: str, *, slug: str, title: str, body: str
) -> models.Post:
    repo = PostRepository(session, tenant_id)
    if repo.get_by_slug(slug) is not None:
        raise ConflictError(
            f"a post with slug {slug!r} already exists in this tenant",
            details={"slug": slug},
        )
    post = repo.add(models.Post(slug=slug, title=title, body=body))
    session.flush()
    return post


def list_posts(
    session: Session, tenant_id: str, *, limit: int = 100, offset: int = 0
) -> List[models.Post]:
    return PostRepository(session, tenant_id).list_page(limit=limit, offset=offset)


def resolve_post(session: Session, tenant_id: str, post_ref: str) -> models.Post:
    """Accept either an id or a slug — slugs make curl transcripts readable."""
    repo = PostRepository(session, tenant_id)
    post = repo.get(post_ref) or repo.get_by_slug(post_ref)
    if post is None:
        raise NotFoundError(
            f"post {post_ref!r} not found (looked up by id and slug)"
        )
    return post
