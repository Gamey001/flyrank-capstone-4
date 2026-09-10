"""Image + tag persistence."""

from typing import List, Optional, Sequence

from sqlalchemy import select, update

from app.db import models
from app.repositories.base import TenantRepository


class ImageRepository(TenantRepository[models.Image]):
    model = models.Image

    def get_by_sha(self, sha256: str) -> Optional[models.Image]:
        return self.session.scalars(
            self._scoped().where(models.Image.sha256 == sha256)
        ).first()

    def get_by_filename(self, filename: str) -> Optional[models.Image]:
        return self.session.scalars(
            self._scoped().where(models.Image.filename == filename)
        ).first()

    def get_by_slug(self, slug: str) -> Optional[models.Image]:
        """Look an image up by filename stem.

        The eval set and the corpus manifest identify images by slug, because
        the extension differs between a downloaded corpus (.jpg) and the
        offline placeholder corpus (.png).
        """
        for image in self.list_all():
            if image.filename.rsplit(".", 1)[0] == slug:
                return image
        return None

    def list_filtered(
        self,
        *,
        status: Optional[str] = None,
        category: Optional[str] = None,
        needs_review: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[models.Image]:
        stmt = self._scoped()
        if status:
            stmt = stmt.where(models.Image.status == status)
        if category:
            stmt = stmt.where(models.Image.category == category)
        if needs_review is not None:
            stmt = stmt.where(models.Image.needs_review.is_(needs_review))
        stmt = stmt.order_by(models.Image.filename).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def list_pending(self, limit: Optional[int] = None) -> List[models.Image]:
        """Images the vision batch job still has to process."""
        stmt = (
            self._scoped()
            .where(models.Image.status.in_([models.IMAGE_PENDING, models.IMAGE_FAILED]))
            .order_by(models.Image.filename)
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def list_tagged(self) -> List[models.Image]:
        return list(
            self.session.scalars(
                self._scoped()
                .where(models.Image.status == models.IMAGE_TAGGED)
                .order_by(models.Image.filename)
            )
        )

    def replace_tags(self, image: models.Image, tags: Sequence[tuple]) -> None:
        """Replace an image's normalised tags with ``(kind, value)`` pairs."""
        self.session.query(models.ImageTag).filter(
            models.ImageTag.image_id == image.id
        ).delete(synchronize_session=False)
        seen = set()
        for kind, value in tags:
            key = (kind, value)
            if not value or key in seen:
                continue
            seen.add(key)
            self.session.add(
                models.ImageTag(
                    tenant_id=self.tenant_id,
                    image_id=image.id,
                    kind=kind,
                    value=value,
                )
            )

    def reset_status(self, image_ids: Sequence[str]) -> int:
        """Force re-tagging of specific images (used by the re-run job)."""
        if not image_ids:
            return 0
        result = self.session.execute(
            update(models.Image)
            .where(
                models.Image.tenant_id == self.tenant_id,
                models.Image.id.in_(image_ids),
            )
            .values(status=models.IMAGE_PENDING)
        )
        return int(result.rowcount or 0)

    def counts_by_status(self) -> dict:
        from sqlalchemy import func

        rows = self.session.execute(
            select(models.Image.status, func.count())
            .where(models.Image.tenant_id == self.tenant_id)
            .group_by(models.Image.status)
        ).all()
        return {status: int(count) for status, count in rows}
