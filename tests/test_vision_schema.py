"""Requirement: vision output is schema-validated; invalid is never trusted."""

import json

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.errors import SchemaValidationError
from app.schemas.vision import ImageTags


def test_valid_payload_is_normalised():
    tags = ImageTags.model_validate(
        {
            "subject": "  Red Fox ",
            "category": "ANIMAL",
            "attributes": ["Snow", "snow", " wild "],
            "caption": "A red fox in the snow",
            "confidence": 0.91,
        }
    )
    assert tags.subject == "red fox"
    assert tags.category == "animal"
    # duplicates collapse, case and whitespace normalise
    assert tags.attributes == ["snow", "wild"]


@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"category": "animal", "caption": "a fox here", "confidence": 0.9},
         "missing subject"),
        ({"subject": "fox", "category": "spaceship", "caption": "a fox here",
          "confidence": 0.9}, "unknown category"),
        ({"subject": "fox", "category": "animal", "caption": "a fox here",
          "confidence": 1.7}, "confidence out of range"),
        ({"subject": "fox", "category": "animal", "caption": "no",
          "confidence": 0.9}, "caption too short"),
        ({"subject": "fox", "category": "animal", "caption": "a fox here",
          "confidence": "very sure"}, "confidence not a number"),
    ],
)
def test_invalid_payloads_are_rejected(payload, reason):
    with pytest.raises(PydanticValidationError):
        ImageTags.model_validate(payload)


def test_prose_instead_of_json_is_a_provider_error(monkeypatch, tmp_path):
    """A 200 OK carrying prose is a failure, not data."""
    import httpx

    from app.providers.gemini import GeminiVisionProvider

    image = tmp_path / "red-fox-snow-01.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Sure! It looks like a fox."}]}}
                ]
            },
        )

    provider = GeminiVisionProvider(
        api_key="test-key",
        model="gemini-2.0-flash",
        base_url="https://example.invalid/v1beta",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SchemaValidationError):
        provider.describe_image(str(image))


def test_schema_violating_json_is_a_provider_error(tmp_path):
    """Well-formed JSON that breaks the contract is still refused."""
    import httpx

    from app.providers.gemini import GeminiVisionProvider

    image = tmp_path / "red-fox-snow-01.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")

    bad = json.dumps(
        {
            "subject": "red fox",
            "category": "animal",
            "attributes": [],
            "caption": "A red fox",
            "confidence": 4.2,  # out of range
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": bad}]}}]}
        )

    provider = GeminiVisionProvider(
        api_key="test-key",
        model="gemini-2.0-flash",
        base_url="https://example.invalid/v1beta",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SchemaValidationError):
        provider.describe_image(str(image))


def test_api_key_is_sent_in_a_header_not_the_url(tmp_path):
    """A key in a query string ends up in access logs. It must not be there."""
    import httpx

    from app.providers.gemini import GeminiVisionProvider

    image = tmp_path / "red-fox-snow-01.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-goog-api-key")
        payload = json.dumps(
            {
                "subject": "red fox",
                "category": "animal",
                "attributes": ["snow"],
                "caption": "A red fox in the snow",
                "confidence": 0.9,
            }
        )
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": payload}]}}]}
        )

    provider = GeminiVisionProvider(
        api_key="super-secret-key",
        model="gemini-2.0-flash",
        base_url="https://example.invalid/v1beta",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.describe_image(str(image))

    assert result.tags.subject == "red fox"
    assert seen["header"] == "super-secret-key"
    assert "super-secret-key" not in seen["url"]
