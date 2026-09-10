"""Repository base.

Every repository is constructed with a tenant and *cannot* be used without
one — tenant isolation is a property of the data-access layer, not something
each endpoint has to remember (shared requirement #4).
"""

from typing import Generic, Iterable, List, Optional, Type, TypeVar

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class TenantRepository(Generic[ModelT]):
    model: Type[ModelT]

    def __init__(self, session: Session, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for every repository")
        self.session = session
        self.tenant_id = tenant_id

    # --- query helpers ----------------------------------------------------
    def _scoped(self) -> Select:
        return select(self.model).where(self.model.tenant_id == self.tenant_id)

    def get(self, entity_id: str) -> Optional[ModelT]:
        return self.session.scalars(
            self._scoped().where(self.model.id == entity_id)
        ).first()

    def list_all(self) -> List[ModelT]:
        return list(self.session.scalars(self._scoped()))

    def count(self) -> int:
        from sqlalchemy import func

        return int(
            self.session.scalar(
                select(func.count())
                .select_from(self.model)
                .where(self.model.tenant_id == self.tenant_id)
            )
            or 0
        )

    # --- writes -----------------------------------------------------------
    def add(self, entity: ModelT) -> ModelT:
        entity.tenant_id = self.tenant_id
        self.session.add(entity)
        return entity

    def add_all(self, entities: Iterable[ModelT]) -> None:
        for entity in entities:
            self.add(entity)

    def flush(self) -> None:
        self.session.flush()
