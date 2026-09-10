"""Evaluation endpoint — the same code path the CLI eval script uses."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db, get_tenant_id
from app.core.config import Settings
from app.services.evaluation import run_evaluation

router = APIRouter(prefix="/eval", tags=["evaluation"])

DEFAULT_EVAL_SET = "data/seed/eval_set.json"


@router.post(
    "/run",
    summary="Score the labeled eval set and report top-1 precision",
    response_description=(
        "Top-1 precision over the labeled cases plus the correct-rejection "
        "rate over the cases whose right answer is a refusal."
    ),
)
def run_eval(
    session: Session = Depends(get_db),
    tenant_id: str = Depends(get_tenant_id),
    settings: Settings = Depends(get_app_settings),
    eval_set: str = DEFAULT_EVAL_SET,
) -> dict:
    report = run_evaluation(session, tenant_id, settings, eval_set)
    return report.to_dict()
