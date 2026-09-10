"""Score the labeled eval set and report top-1 precision (PROBE 5).

    python -m scripts.run_eval
    python -m scripts.run_eval --json eval/report.json

The number this prints is the number in the README. If they disagree, the
README is wrong.
"""

import argparse
import json
import os
import sys
from typing import List

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_scope
from app.services.evaluation import run_evaluation

DEFAULT_EVAL_SET = "data/seed/eval_set.json"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Also write the full report to this path.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING" if args.quiet else settings.log_level)
    tenant_id = args.tenant or settings.default_tenant

    with session_scope() as session:
        report = run_evaluation(session, tenant_id, settings, args.eval_set)

    width = max(len(c.post_slug) for c in report.cases) + 2
    print("\nCASE RESULTS")
    print("-" * (width + 62))
    for case in report.cases:
        mark = "PASS" if case.correct else "FAIL"
        expected = case.expected_image or "(no match expected)"
        predicted = case.predicted_image or "(refused)"
        score = (
            f"{case.predicted_similarity:.3f}"
            if case.predicted_similarity is not None
            else "  -  "
        )
        print(
            f"{mark}  {case.post_slug:<{width}} expected={expected:<22} "
            f"got={predicted:<22} sim={score}"
        )
        if not case.correct and case.rejection_reasons:
            print(f"{'':>6}{'':<{width}} reasons: {', '.join(case.rejection_reasons)}")

    print("-" * (width + 62))
    print(f"embedding model      : {report.embedding_model}")
    print(f"similarity threshold : {report.thresholds['similarity']}")
    print(
        f"TOP-1 PRECISION      : {report.top1_precision:.3f}  "
        f"({report.labeled_correct}/{report.labeled_cases} labeled posts)"
    )
    print(
        f"CORRECT REJECTION    : {report.correct_rejection_rate:.3f}  "
        f"({report.no_match_correct}/{report.no_match_cases} no-match posts)"
    )

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or ".", exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
            fh.write("\n")
        print(f"\nfull report → {args.json_path}")

    # Non-zero exit if the system regressed below the documented number, so
    # this can run in CI as a quality gate.
    return 0 if report.top1_precision >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
