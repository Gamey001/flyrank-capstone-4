"""Vector storage.

At ~50 images the "index" is a full scan of a JSON column, which is both fast
enough (sub-millisecond) and honest — see the README's scaling note for what
would change at 10k images.
"""

from typing import Dict, List, Optional, Sequence

from app.db import models
from app.repositories.base import TenantRepository

OWNER_IMAGE = "image"
OWNER_POST = "post"


class EmbeddingRepository(TenantRepository[models.Embedding]):
    model = models.Embedding

    def get_for(
        self, owner_type: str, owner_id: str, embedding_model: str
    ) -> Optional[models.Embedding]:
        return self.session.scalars(
            self._scoped().where(
                models.Embedding.owner_type == owner_type,
                models.Embedding.owner_id == owner_id,
                models.Embedding.model == embedding_model,
            )
        ).first()

    def upsert(
        self,
        *,
        owner_type: str,
        owner_id: str,
        embedding_model: str,
        vector: Sequence[float],
        source_text: str,
    ) -> models.Embedding:
        """Idempotent: re-embedding the same owner overwrites its vector."""
        existing = self.get_for(owner_type, owner_id, embedding_model)
        if existing is not None:
            existing.vector = list(vector)
            existing.dim = len(vector)
            existing.source_text = source_text
            return existing
        return self.add(
            models.Embedding(
                owner_type=owner_type,
                owner_id=owner_id,
                model=embedding_model,
                dim=len(vector),
                vector=list(vector),
                source_text=source_text,
            )
        )

    def list_by_owner_type(
        self, owner_type: str, embedding_model: str
    ) -> List[models.Embedding]:
        return list(
            self.session.scalars(
                self._scoped().where(
                    models.Embedding.owner_type == owner_type,
                    models.Embedding.model == embedding_model,
                )
            )
        )

    def vectors_by_owner(
        self, owner_type: str, embedding_model: str
    ) -> Dict[str, List[float]]:
        return {
            row.owner_id: row.vector
            for row in self.list_by_owner_type(owner_type, embedding_model)
        }

    def missing_owner_ids(
        self, owner_type: str, owner_ids: Sequence[str], embedding_model: str
    ) -> List[str]:
        have = {
            row.owner_id
            for row in self.list_by_owner_type(owner_type, embedding_model)
        }
        return [oid for oid in owner_ids if oid not in have]
