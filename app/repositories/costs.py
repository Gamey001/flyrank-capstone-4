"""AI cost ledger persistence."""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func, select

from app.db import models
from app.db.base import utcnow
from app.repositories.base import TenantRepository


class AiCallRepository(TenantRepository[models.AiCall]):
    model = models.AiCall

    def record(self, **fields) -> models.AiCall:
        return self.add(models.AiCall(**fields))

    def list_recent(
        self, limit: int = 100, operation: Optional[str] = None
    ) -> List[models.AiCall]:
        stmt = self._scoped().order_by(models.AiCall.created_at.desc()).limit(limit)
        if operation:
            stmt = stmt.where(models.AiCall.operation == operation)
        return list(self.session.scalars(stmt))

    def spend_since(self, since: datetime) -> float:
        total = self.session.scalar(
            select(func.coalesce(func.sum(models.AiCall.cost_usd), 0.0)).where(
                models.AiCall.tenant_id == self.tenant_id,
                models.AiCall.created_at >= since,
            )
        )
        return float(total or 0.0)

    def spend_today(self) -> float:
        return self.spend_since(utcnow() - timedelta(days=1))

    def summary(self) -> dict:
        rows = self.session.execute(
            select(
                models.AiCall.operation,
                models.AiCall.provider,
                models.AiCall.model,
                func.count(),
                func.coalesce(func.sum(models.AiCall.cost_usd), 0.0),
                func.coalesce(func.sum(models.AiCall.input_tokens), 0),
                func.coalesce(func.sum(models.AiCall.output_tokens), 0),
            )
            .where(models.AiCall.tenant_id == self.tenant_id)
            .group_by(
                models.AiCall.operation, models.AiCall.provider, models.AiCall.model
            )
            .order_by(models.AiCall.operation)
        ).all()
        breakdown = [
            {
                "operation": op,
                "provider": provider,
                "model": model,
                "calls": int(calls),
                "cost_usd": round(float(cost), 6),
                "input_tokens": int(inp),
                "output_tokens": int(out),
            }
            for op, provider, model, calls, cost, inp, out in rows
        ]
        errors = int(
            self.session.scalar(
                select(func.count())
                .select_from(models.AiCall)
                .where(
                    models.AiCall.tenant_id == self.tenant_id,
                    models.AiCall.status == "error",
                )
            )
            or 0
        )
        return {
            "total_calls": sum(b["calls"] for b in breakdown),
            "total_cost_usd": round(sum(b["cost_usd"] for b in breakdown), 6),
            "failed_calls": errors,
            "by_operation": breakdown,
        }

