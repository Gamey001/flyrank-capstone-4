"""Offline evaluation.

Top-1 precision over a hand-labeled set is the headline quality number, and it
is the only thing that makes a threshold defensible: "0.62" is not a feeling,
it is the value that maximises this number.

Two populations are scored separately because they measure different things:

* **labeled cases** — a human named the single correct image. Top-1 precision
  is the share whose first *accepted* suggestion is that image.
* **no-match cases** — the correct behaviour is a refusal. Scored as correct
  rejection rate; counting these as precision failures would reward a system
  that guesses.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ValidationError
from app.providers.registry import build_embedding_provider
from app.repositories.images import ImageRepository
from app.repositories.posts import PostRepository
from app.services.costs import CostTracker
from app.services.embedding_service import EmbeddingService
from app.services.matching import MatchingService
from app.core.logging import safe_extra

logger = logging.getLogger("app.eval")


def _slug(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


@dataclass
class CaseResult:
    post_slug: str
    post_title: str
    expected_image: Optional[str]
    predicted_image: Optional[str]
    predicted_similarity: Optional[float]
    correct: bool
    kind: str  # "labeled" | "no_match"
    note: str = ""
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    embedding_model: str
    thresholds: Dict[str, float]
    labeled_cases: int
    labeled_correct: int
    top1_precision: float
    no_match_cases: int
    no_match_correct: int
    correct_rejection_rate: float
    cases: List[CaseResult]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cases"] = [asdict(c) if not isinstance(c, dict) else c
                         for c in self.cases]
        return data

    def summary_line(self) -> str:
        return (
            f"top-1 precision {self.top1_precision:.3f} "
            f"({self.labeled_correct}/{self.labeled_cases}) · "
            f"correct rejection {self.correct_rejection_rate:.3f} "
            f"({self.no_match_correct}/{self.no_match_cases})"
        )


def load_eval_set(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise ValidationError(
            f"eval set not found at {path!r}", details={"path": path}
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data.get("cases", [])
    if not cases:
        raise ValidationError(f"eval set at {path!r} contains no cases")
    return cases


def run_evaluation(
    session: Session,
    tenant_id: str,
    settings: Settings,
    eval_set_path: str,
    *,
    overrides: Optional[Dict[str, float]] = None,
) -> EvalReport:
    """Score the labeled set with the current (or overridden) thresholds."""
    effective = settings
    if overrides:
        effective = settings.model_copy(update=overrides)

    embedding_service = EmbeddingService(
        session,
        tenant_id,
        effective,
        build_embedding_provider(effective),
        CostTracker(session, tenant_id, effective),
    )
    matching = MatchingService(session, tenant_id, effective, embedding_service)
    posts_repo = PostRepository(session, tenant_id)
    images_repo = ImageRepository(session, tenant_id)

    results: List[CaseResult] = []
    for case in load_eval_set(eval_set_path):
        slug = case["post_slug"]
        post = posts_repo.get_by_slug(slug)
        if post is None:
            raise ValidationError(
                f"eval set references unknown post slug {slug!r}; run the seed "
                "script first",
                details={"post_slug": slug},
            )
        expected_slug = case.get("expected_image")
        result = matching.match_post(post)
        accepted = result.accepted
        top = accepted[0] if accepted else None

        predicted_slug = _slug(top.image.filename) if top else None
        if expected_slug:
            if images_repo.get_by_slug(expected_slug) is None:
                raise ValidationError(
                    f"eval set references unknown image {expected_slug!r}",
                    details={"expected_image": expected_slug},
                )
            correct = predicted_slug == expected_slug
            kind = "labeled"
        else:
            correct = top is None
            kind = "no_match"

        results.append(
            CaseResult(
                post_slug=slug,
                post_title=post.title,
                expected_image=expected_slug,
                predicted_image=predicted_slug,
                predicted_similarity=round(top.similarity, 4) if top else None,
                correct=correct,
                kind=kind,
                note=case.get("note", ""),
                rejection_reasons=(
                    []
                    if top
                    else sorted(
                        {
                            reason.code
                            for candidate in result.candidates
                            for reason in candidate.verdict.reasons
                        }
                    )
                ),
            )
        )

    labeled = [r for r in results if r.kind == "labeled"]
    no_match = [r for r in results if r.kind == "no_match"]
    labeled_correct = sum(1 for r in labeled if r.correct)
    no_match_correct = sum(1 for r in no_match if r.correct)

    report = EvalReport(
        embedding_model=embedding_service.model_name,
        thresholds={
            "similarity": effective.similarity_threshold,
            "min_vision_confidence": effective.min_vision_confidence,
            "low_confidence_threshold": effective.low_confidence_threshold,
        },
        labeled_cases=len(labeled),
        labeled_correct=labeled_correct,
        top1_precision=round(labeled_correct / len(labeled), 4) if labeled else 0.0,
        no_match_cases=len(no_match),
        no_match_correct=no_match_correct,
        correct_rejection_rate=(
            round(no_match_correct / len(no_match), 4) if no_match else 0.0
        ),
        cases=results,
    )
    logger.info(
        "evaluation_complete",
        extra=safe_extra({"summary": report.summary_line()}),
    )
    return report
