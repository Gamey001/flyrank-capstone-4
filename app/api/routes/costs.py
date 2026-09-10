"""The AI cost ledger."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_cost_tracker, get_db, get_tenant_id
from app.repositories.costs import AiCallRepository
from app.schemas.api import AiCallOut, CostSummaryOut
from app.services.costs import CostTracker

router = APIRouter(prefix="/costs", tags=["costs"])


@router.get(
    "/summary",
    response_model=CostSummaryOut,
    summary="Spend by operation, plus remaining daily budget",
)
def cost_summary(tracker: CostTracker = Depends(get_cost_tracker)) -> CostSummaryOut:
    return CostSummaryOut(**tracker.summary())


@router.get(
    "",
    response_model=List[AiCallOut],
    summary="The per-call cost log",
    response_description="Newest first; one row per AI call, successes and failures.",
)
def list_calls(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    operation: Optional[str] = Query(
        default=None, description="vision | embedding"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
) -> List[AiCallOut]:
    rows = AiCallRepository(session, tenant_id).list_recent(
        limit=limit, operation=operation
    )
    return [AiCallOut.model_validate(row) for row in rows]
