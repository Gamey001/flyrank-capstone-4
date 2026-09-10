"""Post persistence."""

from typing import List, Optional

from app.db import models
from app.repositories.base import TenantRepository


class PostRepository(TenantRepository[models.Post]):
    model = models.Post

    def get_by_slug(self, slug: str) -> Optional[models.Post]:
        return self.session.scalars(
            self._scoped().where(models.Post.slug == slug)
        ).first()

    def list_page(self, limit: int = 100, offset: int = 0) -> List[models.Post]:
        return list(
            self.session.scalars(
                self._scoped().order_by(models.Post.slug).limit(limit).offset(offset)
            )
        )
