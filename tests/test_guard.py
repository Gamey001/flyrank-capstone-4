"""The mismatch guard — the decision core.

These tests are the executable version of the capstone's headline promise:
the wolf never gets suggested for the fox post, and the refusal says why.
"""

import pytest

from app.services import guard as g

THRESHOLDS = g.GuardThresholds(
    similarity=0.62, min_confidence=0.55, low_confidence_flag=0.70
)

FOX_POST = g.PostIntent.from_text(
    "p1", "The behaviour of red foxes", "Red foxes hunt at dusk in the forest."
)


def candidate(**overrides) -> g.Candidate:
    base = dict(
        image_id="img-1",
        subject="red fox",
        category="animal",
        caption="A red fox standing in a forest",
        confidence=0.94,
    )
    base.update(overrides)
    return g.Candidate(**base)


def test_matching_subject_is_accepted():
    verdict = g.evaluate(FOX_POST, candidate(), 0.88, THRESHOLDS)
    assert verdict.accepted
    assert verdict.primary_reason.code == g.REASON_OK


def test_wolf_is_rejected_for_a_fox_post_even_at_high_similarity():
    """PROBE 3. The wolf caption is semantically *close* — that is the point."""
    wolf = candidate(
        image_id="img-wolf", subject="gray wolf", caption="A gray wolf in the forest"
    )
    verdict = g.evaluate(FOX_POST, wolf, 0.94, THRESHOLDS)

    assert not verdict.accepted
    assert verdict.primary_reason.code == g.REASON_SUBJECT_MISMATCH
    message = verdict.primary_reason.message.lower()
    assert "mismatch" in message
    assert "red fox" in message and "gray wolf" in message


def test_dog_is_also_rejected_for_a_fox_post():
    dog = candidate(image_id="img-dog", subject="dog", caption="A dog in a park")
    verdict = g.evaluate(FOX_POST, dog, 0.80, THRESHOLDS)
    assert not verdict.accepted
    assert verdict.primary_reason.code == g.REASON_SUBJECT_MISMATCH


def test_unrelated_category_is_rejected_with_a_category_reason():
    pizza = candidate(
        image_id="img-pizza",
        subject="pizza",
        category="food",
        caption="A margherita pizza",
    )
    verdict = g.evaluate(FOX_POST, pizza, 0.70, THRESHOLDS)
    assert not verdict.accepted
    assert verdict.primary_reason.code in {
        g.REASON_CATEGORY_MISMATCH,
        g.REASON_SUBJECT_MISMATCH,
    }


def test_low_similarity_is_rejected_even_with_the_right_subject():
    verdict = g.evaluate(FOX_POST, candidate(), 0.30, THRESHOLDS)
    assert not verdict.accepted
    assert any(r.code == g.REASON_BELOW_THRESHOLD for r in verdict.reasons)
    assert "0.30" in verdict.primary_reason.message


def test_low_vision_confidence_is_rejected():
    unsure = candidate(confidence=0.41)
    verdict = g.evaluate(FOX_POST, unsure, 0.95, THRESHOLDS)
    assert not verdict.accepted
    assert verdict.primary_reason.code == g.REASON_LOW_CONFIDENCE


def test_flagged_image_is_not_auto_approved():
    flagged = candidate(confidence=0.60, needs_review=True)
    verdict = g.evaluate(FOX_POST, flagged, 0.95, THRESHOLDS)
    assert not verdict.accepted
    assert verdict.primary_reason.code == g.REASON_NEEDS_REVIEW


def test_untagged_image_is_never_suggested():
    untagged = g.Candidate(
        image_id="img-x",
        subject=None,
        category=None,
        caption="",
        confidence=None,
        is_tagged=False,
    )
    verdict = g.evaluate(FOX_POST, untagged, 0.99, THRESHOLDS)
    assert not verdict.accepted
    assert verdict.primary_reason.code == g.REASON_UNTAGGED


def test_all_failing_reasons_are_reported_not_just_the_first():
    bad = candidate(subject="gray wolf", confidence=0.20)
    verdict = g.evaluate(FOX_POST, bad, 0.10, THRESHOLDS)
    codes = {r.code for r in verdict.reasons}
    assert {
        g.REASON_LOW_CONFIDENCE,
        g.REASON_SUBJECT_MISMATCH,
        g.REASON_BELOW_THRESHOLD,
    } <= codes


def test_ambiguity_is_a_warning_not_a_rejection():
    verdict = g.evaluate(
        FOX_POST, candidate(), 0.90, THRESHOLDS, runner_up_similarity=0.895
    )
    assert verdict.accepted
    assert verdict.signals["ambiguous"] is True


@pytest.mark.parametrize(
    "title, expected",
    [
        ("The behaviour of red foxes", {"red fox"}),
        ("Vulpes vulpes in winter", {"red fox"}),
        ("Canis lupus pack structure", {"gray wolf"}),
        ("Restoring vintage motorcycles", set()),
    ],
)
def test_post_intent_reads_the_subject_from_the_title(title, expected):
    assert g.PostIntent.from_text("p", title, "").subjects == expected


def test_body_does_not_override_a_title_subject():
    """A fox post that mentions wolves once is still a fox post."""
    intent = g.PostIntent.from_text(
        "p", "The red fox in winter", "Unlike the gray wolf, the fox hunts alone."
    )
    assert intent.subjects == {"red fox"}


def test_no_match_explanation_names_the_wanted_subject():
    rejected = [
        g.evaluate(
            FOX_POST,
            candidate(subject="gray wolf", image_id=f"img-{i}"),
            0.4,
            THRESHOLDS,
        )
        for i in range(3)
    ]
    payload = g.no_match_explanation(FOX_POST, rejected, THRESHOLDS)
    assert "red fox" in payload["message"]
    assert payload["candidates_considered"] == 3
    assert payload["best_similarity"] == pytest.approx(0.4)
