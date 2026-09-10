"""Derive the similarity threshold from the labeled eval set.

This is where the guard's number comes from — and the answer is more
interesting than a single maximum.

Three things get printed:

1. **The sweep.** Combined accuracy at each candidate threshold. On this corpus
   it is a wide plateau, because the guard's *tag-agreement* rule already
   refuses wrong-subject candidates before similarity is consulted. The
   similarity threshold is the second line of defence, for posts about
   subjects the corpus does not contain at all.
2. **The safe band.** The lowest similarity among correct matches (the lowest
   score we must still accept) and the highest similarity among posts whose
   right answer is a refusal (the highest score we must still reject). Any
   threshold strictly between them scores perfectly.
3. **The configured value's margin** to each edge of that band, so the value in
   ``.env`` can be defended with two numbers rather than a feeling.

    python -m scripts.tune_thresholds
"""

import argparse
import sys
from typing import List, Optional, Tuple

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_scope
from app.providers.registry import build_embedding_provider
from app.repositories.posts import PostRepository
from app.services.costs import CostTracker
from app.services.embedding_service import EmbeddingService
from app.services.evaluation import load_eval_set, run_evaluation
from app.services.matching import MatchingService

DEFAULT_EVAL_SET = "data/seed/eval_set.json"


def measure_band(
    session, tenant_id: str, settings, eval_set_path: str
) -> Tuple[Optional[float], Optional[float]]:
    """(lowest correct-match similarity, highest must-reject similarity)."""
    embeddings = EmbeddingService(
        session,
        tenant_id,
        settings,
        build_embedding_provider(settings),
        CostTracker(session, tenant_id, settings),
    )
    matching = MatchingService(session, tenant_id, settings, embeddings)
    posts = PostRepository(session, tenant_id)

    positives: List[float] = []
    negatives: List[float] = []
    for case in load_eval_set(eval_set_path):
        post = posts.get_by_slug(case["post_slug"])
        if post is None:
            continue
        result = matching.match_post(post)
        expected = case.get("expected_image")
        if expected:
            for candidate in result.candidates:
                if candidate.image.filename.rsplit(".", 1)[0] == expected:
                    positives.append(candidate.similarity)
                    break
        else:
            negatives.append(
                result.candidates[0].similarity if result.candidates else 0.0
            )
    return (
        min(positives) if positives else None,
        max(negatives) if negatives else None,
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.98)
    parser.add_argument("--step", type=float, default=0.04)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")
    tenant_id = args.tenant or settings.default_tenant

    print(f"{'threshold':>10} {'top-1':>8} {'reject':>8} {'combined':>10}")
    print("-" * 40)

    rows: List[Tuple[float, float]] = []
    with session_scope() as session:
        value = args.start
        while value <= args.stop + 1e-9:
            threshold = round(value, 4)
            report = run_evaluation(
                session,
                tenant_id,
                settings,
                args.eval_set,
                overrides={"similarity_threshold": threshold},
            )
            total = report.labeled_cases + report.no_match_cases
            combined = (
                (report.labeled_correct + report.no_match_correct) / total
                if total
                else 0.0
            )
            rows.append((threshold, combined))
            print(
                f"{threshold:>10.2f} {report.top1_precision:>8.3f} "
                f"{report.correct_rejection_rate:>8.3f} {combined:>10.3f}"
            )
            value += args.step

        lowest_positive, highest_negative = measure_band(
            session, tenant_id, settings, args.eval_set
        )

    print("-" * 40)
    _report_plateau(rows)
    _report_band(lowest_positive, highest_negative, settings.similarity_threshold)
    return 0


def _report_plateau(rows: List[Tuple[float, float]]) -> None:
    if not rows:
        return
    best = max(score for _, score in rows)
    optimal = [threshold for threshold, score in rows if score >= best - 1e-9]
    print(
        f"optimal plateau : {min(optimal):.2f} … {max(optimal):.2f} "
        f"(combined accuracy {best:.3f})"
    )


def _report_band(
    lowest_positive: Optional[float],
    highest_negative: Optional[float],
    configured: float,
) -> None:
    if lowest_positive is None or highest_negative is None:
        print("safe band       : not measurable from this eval set")
        return

    print(
        f"safe band       : {highest_negative:.3f} < threshold < "
        f"{lowest_positive:.3f}"
    )
    print(
        f"  · {lowest_positive:.3f} is the weakest similarity among correct "
        "matches — go above it and the system starts refusing good pairings"
    )
    print(
        f"  · {highest_negative:.3f} is the strongest similarity among posts "
        "that must be refused — go below it and the system starts guessing"
    )
    print(f"  · midpoint (max margin): {(lowest_positive + highest_negative) / 2:.3f}")
    print(
        f"\nconfigured      : {configured:.3f} "
        f"(+{configured - highest_negative:.3f} above the reject edge, "
        f"-{lowest_positive - configured:.3f} below the accept edge)"
    )
    if not highest_negative < configured < lowest_positive:
        print("  ! WARNING: the configured threshold is OUTSIDE the safe band")
    else:
        print(
            "  The default sits above the midpoint on purpose: in this system a "
            "wrong image shipped is worse than a missing one, so the threshold "
            "is biased toward refusing."
        )
    print("\nSet it with SIMILARITY_THRESHOLD in .env, then re-run scripts/run_eval.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
