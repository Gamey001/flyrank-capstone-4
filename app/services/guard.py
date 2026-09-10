"""The mismatch guard.

The production-critical part of this system: given a ranked candidate, decide
whether the recommendation is *actually good enough* — and when it is not, say
why in a sentence a human can act on.

Three independent signals, all of which must pass:

============  ==========================================================
signal        rejects when
============  ==========================================================
confidence    the vision model was not sure enough about what it saw
similarity    the caption and the post are not close enough in meaning
tag agreement the post asks for subject X and the image shows subject Y
============  ==========================================================

Similarity alone is not sufficient, and that is the whole lesson: "a gray wolf
in the forest" and "the behaviour of red foxes" are semantically *close*.
Cosine similarity happily ranks the wolf high; only the tag check knows that
fox and wolf are different animals.

This module is deliberately pure — no database, no I/O, no settings import.
Everything it needs arrives as arguments, which makes each rule directly
unit-testable and makes the thresholds tunable from the eval set.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.services import concepts

# --- reason codes (stable identifiers; the message is the human-readable part)
REASON_LOW_CONFIDENCE = "low_vision_confidence"
REASON_BELOW_THRESHOLD = "similarity_below_threshold"
REASON_SUBJECT_MISMATCH = "subject_mismatch"
REASON_CATEGORY_MISMATCH = "category_mismatch"
REASON_UNTAGGED = "image_not_tagged"
REASON_NEEDS_REVIEW = "image_flagged_for_review"
REASON_AMBIGUOUS = "ambiguous_top_candidates"
REASON_OK = "accepted"


@dataclass(frozen=True)
class GuardThresholds:
    """Tuned on the labeled eval set — see ``scripts/tune_thresholds.py``."""

    similarity: float = 0.62
    min_confidence: float = 0.55
    low_confidence_flag: float = 0.70
    ambiguity_margin: float = 0.03


@dataclass(frozen=True)
class Candidate:
    """The image side of the decision, already validated and tagged."""

    image_id: str
    subject: Optional[str]
    category: Optional[str]
    caption: str
    confidence: Optional[float]
    attributes: Sequence[str] = ()
    needs_review: bool = False
    is_tagged: bool = True

    @property
    def subject_concept(self) -> Optional[str]:
        return concepts.canonical_subject(self.subject or "") or None


@dataclass(frozen=True)
class PostIntent:
    """What the post is asking for, derived once and reused for every candidate."""

    post_id: str
    title: str
    subjects: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    modifiers: Set[str] = field(default_factory=set)

    @classmethod
    def from_text(cls, post_id: str, title: str, body: str = "") -> "PostIntent":
        # The title carries the intent; the body only broadens it. Extracting
        # from both and letting the title win keeps "a post about foxes that
        # mentions wolves once" asking for a fox.
        title_subjects = concepts.extract_subjects(title)
        subjects = title_subjects or concepts.extract_subjects(f"{title} {body}")
        return cls(
            post_id=post_id,
            title=title,
            subjects=subjects,
            categories={
                c for c in (concepts.category_of(s) for s in subjects) if c
            },
            modifiers=concepts.extract_modifiers(f"{title} {body}"),
        )


@dataclass(frozen=True)
class Reason:
    code: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class GuardVerdict:
    accepted: bool
    reasons: List[Reason]
    similarity: float
    #: diagnostics kept verbatim on the suggestion row so /explain is free
    signals: Dict[str, Any]

    @property
    def primary_reason(self) -> Optional[Reason]:
        return self.reasons[0] if self.reasons else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "similarity": round(self.similarity, 4),
            "reasons": [r.as_dict() for r in self.reasons],
            "signals": self.signals,
        }


def evaluate(
    intent: PostIntent,
    candidate: Candidate,
    similarity: float,
    thresholds: GuardThresholds,
    *,
    runner_up_similarity: Optional[float] = None,
) -> GuardVerdict:
    """Decide whether ``candidate`` may be suggested for ``intent``.

    Rules are evaluated in order of how damaging a false accept would be, and
    *all* failing rules are reported — a reviewer should see every reason a
    pairing was refused, not just the first one.
    """
    reasons: List[Reason] = []
    candidate_subject = candidate.subject_concept
    candidate_category = candidate.category or (
        concepts.category_of(candidate_subject) if candidate_subject else None
    )

    signals: Dict[str, Any] = {
        "post_subjects": sorted(intent.subjects),
        "post_modifiers": sorted(intent.modifiers),
        "image_subject": candidate.subject,
        "image_subject_concept": candidate_subject,
        "image_category": candidate_category,
        "image_confidence": candidate.confidence,
        "similarity": round(similarity, 4),
        "thresholds": {
            "similarity": thresholds.similarity,
            "min_confidence": thresholds.min_confidence,
            "ambiguity_margin": thresholds.ambiguity_margin,
        },
    }

    # --- rule 0: we never suggest an image we have not understood ----------
    if not candidate.is_tagged or candidate_subject is None:
        reasons.append(
            Reason(
                REASON_UNTAGGED,
                f"Image {candidate.image_id} has no validated vision metadata, "
                "so it cannot be recommended.",
            )
        )
        return GuardVerdict(False, reasons, similarity, signals)

    # --- rule 1: the model's own uncertainty -------------------------------
    confidence = candidate.confidence if candidate.confidence is not None else 0.0
    if confidence < thresholds.min_confidence:
        reasons.append(
            Reason(
                REASON_LOW_CONFIDENCE,
                f"Vision confidence {confidence:.2f} is below the "
                f"{thresholds.min_confidence:.2f} minimum — the model is not "
                f"sure this image shows a {candidate.subject}.",
            )
        )
    elif candidate.needs_review:
        reasons.append(
            Reason(
                REASON_NEEDS_REVIEW,
                f"Image is flagged for human review "
                f"(confidence {confidence:.2f}) and is not auto-approved.",
            )
        )

    # --- rule 2: subject / category agreement ------------------------------
    # This is the rule cosine similarity cannot express.
    if intent.subjects:
        if candidate_subject not in intent.subjects:
            expected = _humanize(sorted(intent.subjects))
            if any(
                concepts.are_confusable(subject, candidate_subject)
                for subject in intent.subjects
            ):
                reasons.append(
                    Reason(
                        REASON_SUBJECT_MISMATCH,
                        f"{(candidate_category or 'Subject').capitalize()} "
                        f"category mismatch: expected {expected}, "
                        f"detected {candidate_subject}.",
                    )
                )
            elif intent.categories and candidate_category not in intent.categories:
                reasons.append(
                    Reason(
                        REASON_CATEGORY_MISMATCH,
                        f"Post is about {expected} "
                        f"({_humanize(sorted(intent.categories))}), but the image "
                        f"shows {candidate_subject} ({candidate_category}).",
                    )
                )
            else:
                reasons.append(
                    Reason(
                        REASON_SUBJECT_MISMATCH,
                        f"Post asks for {expected}, but the image shows "
                        f"{candidate_subject}.",
                    )
                )

    # --- rule 3: semantic distance ----------------------------------------
    if similarity < thresholds.similarity:
        reasons.append(
            Reason(
                REASON_BELOW_THRESHOLD,
                f"Similarity {similarity:.2f} is below the "
                f"{thresholds.similarity:.2f} threshold — the image caption "
                f"is not close enough in meaning to this post.",
            )
        )

    # --- rule 4: ambiguity is a warning, not a rejection --------------------
    if runner_up_similarity is not None:
        margin = similarity - runner_up_similarity
        signals["margin_to_runner_up"] = round(margin, 4)
        if not reasons and margin < thresholds.ambiguity_margin:
            signals["ambiguous"] = True
            signals["ambiguity_note"] = (
                f"Top two candidates are within {margin:.3f}; the pick is "
                "accepted but worth a human glance."
            )

    if reasons:
        return GuardVerdict(False, reasons, similarity, signals)

    return GuardVerdict(
        True,
        [
            Reason(
                REASON_OK,
                f"Subject {candidate_subject} matches the post, similarity "
                f"{similarity:.2f} ≥ {thresholds.similarity:.2f}, vision "
                f"confidence {confidence:.2f}.",
            )
        ],
        similarity,
        signals,
    )


def no_match_explanation(
    intent: PostIntent,
    rejected: Sequence[GuardVerdict],
    thresholds: GuardThresholds,
) -> Dict[str, Any]:
    """Explain a "no confident match" answer for a whole post.

    Aggregates *why* each candidate failed so the answer is actionable
    ("nothing in the corpus is a bear") rather than a bare empty list.
    """
    counts: Dict[str, int] = {}
    for verdict in rejected:
        for reason in verdict.reasons:
            counts[reason.code] = counts.get(reason.code, 0) + 1

    best = max((v.similarity for v in rejected), default=0.0)
    wanted = _humanize(sorted(intent.subjects)) if intent.subjects else "this topic"
    parts = [
        f"No image in the corpus clears the bar for “{intent.title}”.",
        f"Best similarity was {best:.2f} against a {thresholds.similarity:.2f} "
        f"threshold.",
    ]
    if counts.get(REASON_SUBJECT_MISMATCH) or counts.get(REASON_CATEGORY_MISMATCH):
        parts.append(f"No image was tagged as {wanted}.")
    if counts.get(REASON_LOW_CONFIDENCE):
        parts.append(
            f"{counts[REASON_LOW_CONFIDENCE]} candidate(s) were rejected for low "
            "vision confidence."
        )
    return {
        "message": " ".join(parts),
        "wanted_subjects": sorted(intent.subjects),
        "best_similarity": round(best, 4),
        "similarity_threshold": thresholds.similarity,
        "rejection_counts": counts,
        "candidates_considered": len(rejected),
    }


def _humanize(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return "an unspecified subject"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"
