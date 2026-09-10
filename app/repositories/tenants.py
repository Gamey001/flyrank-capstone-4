"""Tenant lookup — the only repository that is not itself tenant-scoped."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models


def get_tenant(session: Session, tenant_id: str) -> Optional[models.Tenant]:
    return session.scalars(
        select(models.Tenant).where(models.Tenant.id == tenant_id)
    ).first()


def ensure_tenant(session: Session, tenant_id: str, name: str = "") -> models.Tenant:
    """Create the tenant if missing — used by seeding, never by the API."""
    tenant = get_tenant(session, tenant_id)
    if tenant is None:
        tenant = models.Tenant(id=tenant_id, name=name or tenant_id)
        session.add(tenant)
        session.flush()
    return tenant
