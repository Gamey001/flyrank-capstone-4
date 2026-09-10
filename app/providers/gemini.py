"""Gemini Flash providers (free tier, Google account, no credit card).

Called over plain HTTP with ``httpx`` rather than the vendor SDK: two small
endpoints, no extra dependency, and a request shape that is easy to assert on
in tests.
"""

import base64
import json
import mimetypes
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.core.errors import ProviderError, SchemaValidationError
from app.providers.base import EmbeddingResult, Usage, VisionResult
from app.schemas.vision import IMAGE_TAGS_JSON_SCHEMA, ImageTags

VISION_PROMPT = (
    "You are tagging a stock photo for an image-matching system.\n"
    "Describe ONLY what is visibly present. Do not speculate.\n"
    "- subject: the single main subject, lowercase (e.g. \"red fox\").\n"
    "- category: one of animal, landscape, food, architecture, other.\n"
    "- attributes: up to 8 short visual descriptors (setting, colour, action).\n"
    "- caption: one factual sentence.\n"
    "- confidence: your honest 0-1 certainty about `subject`. If the animal "
    "could plausibly be a different species, report it BELOW 0.7."
)

#: Retrying a 4xx that is not a rate limit just burns quota.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class GeminiVisionProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set; set it in .env or switch "
                "VISION_PROVIDER to 'ollama' or 'stub'",
                retryable=False,
            )
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def describe_image(
        self, image_path: str, *, hint: Optional[str] = None
    ) -> VisionResult:
        if not os.path.exists(image_path):
            raise ProviderError(f"image not found: {image_path}", retryable=False)
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")

        prompt = VISION_PROMPT if not hint else f"{VISION_PROMPT}\nContext: {hint}"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": data}},
                    ]
                }
            ],
            # Constrain the model at the source; we still validate on arrival.
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(IMAGE_TAGS_JSON_SCHEMA),
            },
        }

        started = time.perf_counter()
        payload = self._post(
            f"/models/{self.model}:generateContent", body
        )
        latency = int((time.perf_counter() - started) * 1000)

        text = _first_text(payload)
        try:
            tags = ImageTags.model_validate_json(text)
        except Exception as exc:
            # A well-formed HTTP 200 carrying malformed tags is still a failure.
            raise SchemaValidationError(
                f"Gemini returned output that does not match the tag schema: {exc}",
                retryable=True,
                details={"raw_text": text[:800]},
            ) from exc

        usage = payload.get("usageMetadata", {})
        return VisionResult(
            tags=tags,
            raw=json.loads(text),
            usage=Usage(
                input_tokens=int(usage.get("promptTokenCount", 0)),
                output_tokens=int(usage.get("candidatesTokenCount", 0)),
                latency_ms=latency,
                meta={"finish_reason": _finish_reason(payload)},
            ),
            model=self.model,
        )

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return _gemini_post(
            self._client, self._base_url, path, body, self._api_key
        )


class GeminiEmbeddingProvider:
    name = "gemini"
    #: text-embedding-004 output width
    dim = 768

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set; set it in .env or switch "
                "EMBEDDING_PROVIDER to 'ollama' or 'stub'",
                retryable=False,
            )
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def embed(self, text: str) -> EmbeddingResult:
        started = time.perf_counter()
        payload = _gemini_post(
            self._client,
            self._base_url,
            f"/models/{self.model}:embedContent",
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                # Captions and post bodies must land in one comparable space.
                "taskType": "SEMANTIC_SIMILARITY",
            },
            self._api_key,
        )
        latency = int((time.perf_counter() - started) * 1000)
        values = payload.get("embedding", {}).get("values")
        if not isinstance(values, list) or not values:
            raise SchemaValidationError(
                "Gemini embedding response had no 'embedding.values' array",
                details={"payload_keys": list(payload)},
            )
        return EmbeddingResult(
            vector=[float(v) for v in values],
            usage=Usage(
                input_tokens=max(1, len(text.split())),
                latency_ms=latency,
            ),
            model=self.model,
        )

    def embed_many(self, texts: Sequence[str]) -> List[EmbeddingResult]:
        return [self.embed(t) for t in texts]


# --- shared HTTP plumbing ---------------------------------------------------
def _gemini_post(
    client: httpx.Client,
    base_url: str,
    path: str,
    body: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    try:
        response = client.post(
            f"{base_url}{path}",
            json=body,
            headers={
                "x-goog-api-key": api_key,  # never in the URL: URLs get logged
                "content-type": "application/json",
            },
        )
    except httpx.RequestError as exc:
        raise ProviderError(f"Gemini request failed: {exc}", retryable=True) from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"Gemini returned HTTP {response.status_code}: {response.text[:300]}",
            retryable=response.status_code in _RETRYABLE_STATUS,
            details={"status_code": response.status_code},
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(
            "Gemini returned a non-JSON body", retryable=True
        ) from exc


def _first_text(payload: Dict[str, Any]) -> str:
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SchemaValidationError(
            "Gemini response contained no text part",
            details={"finish_reason": _finish_reason(payload)},
        ) from exc


def _finish_reason(payload: Dict[str, Any]) -> Optional[str]:
    try:
        return payload["candidates"][0].get("finishReason")
    except (KeyError, IndexError, TypeError):
        return None


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """JSON Schema → the OpenAPI subset Gemini's ``responseSchema`` accepts."""
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"minLength", "maxLength", "minimum", "maximum", "maxItems"}:
            continue
        if key == "type":
            out["type"] = value.upper()
        elif key == "properties":
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        else:
            out[key] = value
    return out
