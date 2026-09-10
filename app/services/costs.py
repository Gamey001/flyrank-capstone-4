"""Cost tracking and the budget guard (shared requirement #7).

Every AI call is written to ``ai_calls`` — successes *and* failures, each
attributed to the image or post it was made for and to the job that made it.
Before a call is issued the daily spend is checked against
``AI_DAILY_BUDGET_USD``; over budget, the call is refused rather than made.

The free tier bills $0. We still price every call at the published paid rate,
because a cost log that always reads zero teaches nobody anything and would not
catch the day someone flips to a paid model.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import BudgetExceededError
from app.repositories.costs import AiCallRepository
from app.core.logging import safe_extra

logger = logging.getLogger("app.costs")

OP_VISION = "vision"
OP_EMBEDDING = "embedding"


class CostTracker:
    """Writes the ledger and enforces the budget for one tenant."""

    def __init__(self, session: Session, tenant_id: str, settings: Settings):
        self.repo = AiCallRepository(session, tenant_id)
        self.settings = settings

    # --- pricing ---------------------------------------------------------
    def price(
        self, operation: str, *, input_tokens: int = 0, output_tokens: int = 0
    ) -> float:
        if operation == OP_VISION:
            # Vision is billed per call in our model: one image, one short
            # JSON answer. Simple, and it makes the per-image cost obvious.
            return self.settings.vision_cost_per_call_usd
        total_tokens = input_tokens + output_tokens
        return (
            total_tokens / 1000.0
        ) * self.settings.embedding_cost_per_1k_tokens_usd

    # --- budget guard ----------------------------------------------------
    def spend_today(self) -> float:
        return self.repo.spend_today()

    def assert_within_budget(self, projected_cost: float = 0.0) -> None:
        budget = self.settings.ai_daily_budget_usd
        if budget <= 0:
            return
        spent = self.spend_today()
        if spent + projected_cost > budget:
            raise BudgetExceededError(
                f"Daily AI budget exhausted: ${spent:.4f} spent of ${budget:.4f}; "
                f"this call would add ${projected_cost:.4f}. Raise "
                "AI_DAILY_BUDGET_USD or wait for the window to roll.",
                details={
                    "spent_usd": round(spent, 6),
                    "budget_usd": budget,
                    "projected_cost_usd": round(projected_cost, 6),
                },
            )

    # --- ledger ----------------------------------------------------------
    def record(
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        subject_ref: str,
        job_id: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        status: str = "ok",
        error: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> float:
        # A failed call still consumed quota and latency; it is logged at $0
        # for spend but stays visible in the ledger.
        cost = (
            self.price(
                operation, input_tokens=input_tokens, output_tokens=output_tokens
            )
            if status == "ok"
            else 0.0
        )
        self.repo.record(
            operation=operation,
            provider=provider,
            model=model,
            subject_ref=subject_ref,
            job_id=job_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            status=status,
            error=(error or "")[:2000] or None,
            meta=meta,
        )
        logger.info(
            "ai_call",
            extra=safe_extra({
                "operation": operation,
                "provider": provider,
                "model": model,
                "subject_ref": subject_ref,
                "job_id": job_id,
                "cost_usd": round(cost, 6),
                "latency_ms": latency_ms,
                "status": status,
            }),
        )
        return cost

    def summary(self) -> dict:
        data = self.repo.summary()
        data["daily_budget_usd"] = self.settings.ai_daily_budget_usd
        data["spent_last_24h_usd"] = round(self.spend_today(), 6)
        data["budget_remaining_usd"] = round(
            max(0.0, self.settings.ai_daily_budget_usd - self.spend_today()), 6
        )
        return data
