"""Semantic matching over the real corpus (PROBE 2)."""

import pytest

from app.core.config import get_settings
from app.providers.registry import build_embedding_provider
from app.repositories.images import ImageRepository
from app.repositories.posts import PostRepository
from app.services.costs import CostTracker
from app.services.embedding_service import EmbeddingService
from app.services.matching import MatchingService
from app.services.similarity import cosine_similarity


@pytest.fixture()
def matching(session, seeded, migrated):
    embeddings = EmbeddingService(
        session,
        seeded,
        migrated,
        build_embedding_provider(migrated),
        CostTracker(session, seeded, migrated),
    )
    return MatchingService(session, seeded, migrated, embeddings)


def image_slug(candidate) -> str:
    return candidate.image.filename.rsplit(".", 1)[0]


def test_fox_post_ranks_a_fox_image_first(matching, session, seeded):
    post = PostRepository(session, seeded).get_by_slug("red-foxes-in-deep-snow")
    result = matching.match_post(post)

    assert result.has_match
    assert result.accepted[0].image.subject == "red fox"
    assert image_slug(result.accepted[0]) == "red-fox-snow-01"


def test_wolf_and_dog_rank_below_the_fox_and_are_refused(matching, session, seeded):
    post = PostRepository(session, seeded).get_by_slug("red-foxes-in-deep-snow")
    result = matching.match_post(post, limit=50)

    by_subject = {}
    for candidate in result.candidates:
        by_subject.setdefault(candidate.image.subject, candidate.rank)

    assert by_subject["red fox"] < by_subject["gray wolf"]
    assert by_subject["red fox"] < by_subject["dog"]
    # ranking is not enough — nothing but a fox may be *accepted*
    assert {c.image.subject for c in result.accepted} == {"red fox"}


def test_semantic_match_survives_a_latin_name(matching, session, seeded):
    """"Vulpes vulpes" shares no words with "red fox" — only meaning."""
    post = PostRepository(session, seeded).get_by_slug("the-red-fox-of-the-forest")
    assert "vulpes" in post.body.lower()

    result = matching.match_post(post)
    assert result.has_match
    assert result.accepted[0].image.subject == "red fox"


def test_equivalent_concepts_embed_close_together(migrated):
    provider = build_embedding_provider(migrated)
    fox = provider.embed("red fox").vector
    latin = provider.embed("Vulpes vulpes").vector
    wolf = provider.embed("gray wolf").vector
    pizza = provider.embed("margherita pizza").vector

    assert cosine_similarity(fox, latin) > 0.99      # same concept
    assert cosine_similarity(fox, wolf) < 0.5        # different animal
    assert cosine_similarity(fox, pizza) < 0.2       # different world


def test_post_with_no_suitable_image_gets_an_explained_refusal(
    matching, session, seeded
):
    """PROBE 4."""
    post = PostRepository(session, seeded).get_by_slug("restoring-vintage-motorcycles")
    result = matching.match_post(post)

    assert not result.has_match
    payload = matching.no_match_payload(result)
    assert "No image in the corpus clears the bar" in payload["message"]
    assert payload["similarity_threshold"] == migrated_threshold()
    assert payload["rejection_counts"]


def migrated_threshold() -> float:
    return get_settings().similarity_threshold


def test_forcing_the_wolf_onto_the_fox_post_is_refused(matching, session, seeded):
    """PROBE 3 through the service layer, bypassing ranking entirely."""
    posts = PostRepository(session, seeded)
    images = ImageRepository(session, seeded)
    post = posts.get_by_slug("red-foxes-in-deep-snow")
    wolf = images.get_by_slug("gray-wolf-snow-01")

    candidate = matching.check_pair(post, wolf)
    assert not candidate.verdict.accepted
    assert candidate.verdict.primary_reason.code == "subject_mismatch"
    assert "gray wolf" in candidate.verdict.primary_reason.message


def test_rankings_are_stable_across_runs(matching, session, seeded):
    post = PostRepository(session, seeded).get_by_slug("the-perfect-espresso")
    first = [image_slug(c) for c in matching.match_post(post, limit=10).candidates]
    second = [image_slug(c) for c in matching.match_post(post, limit=10).candidates]
    assert first == second
