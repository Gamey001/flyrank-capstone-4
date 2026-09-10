"""Ollama providers — the fully local, offline, no-key path.

Requires ``ollama serve`` plus ``ollama pull llava`` and
``ollama pull all-minilm``. Same interface as Gemini, so nothing else in the
system changes.
"""

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.core.errors import ProviderError, SchemaValidationError
from app.providers.base import EmbeddingResult, Usage, VisionResult
from app.providers.gemini import VISION_PROMPT
from app.schemas.vision import IMAGE_TAGS_JSON_SCHEMA, ImageTags


class OllamaVisionProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        client: Optional[httpx.Client] = None,
    ):
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def describe_image(
        self, image_path: str, *, hint: Optional[str] = None
    ) -> VisionResult:
        if not os.path.exists(image_path):
            raise ProviderError(f"image not found: {image_path}", retryable=False)
        with open(image_path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")

        started = time.perf_counter()
        payload = self._post(
            "/api/generate",
            {
                "model": self.model,
                "prompt": VISION_PROMPT if not hint else f"{VISION_PROMPT}\n{hint}",
                "images": [data],
                "stream": False,
                # Ollama constrains generation to this JSON Schema.
                "format": IMAGE_TAGS_JSON_SCHEMA,
                "options": {"temperature": 0.1},
            },
        )
        latency = int((time.perf_counter() - started) * 1000)

        text = payload.get("response", "")
        try:
            tags = ImageTags.model_validate_json(text)
        except Exception as exc:
            raise SchemaValidationError(
                f"Ollama returned output that does not match the tag schema: {exc}",
                retryable=True,
                details={"raw_text": text[:800]},
            ) from exc

        return VisionResult(
            tags=tags,
            raw=json.loads(text),
            usage=Usage(
                input_tokens=int(payload.get("prompt_eval_count", 0)),
                output_tokens=int(payload.get("eval_count", 0)),
                latency_ms=latency,
                meta={"local": True},
            ),
            model=self.model,
        )

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return _ollama_post(self._client, self._base_url, path, body)


class OllamaEmbeddingProvider:
    name = "ollama"
    #: all-minilm output width; corrected from the first response either way.
    dim = 384

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def embed(self, text: str) -> EmbeddingResult:
        started = time.perf_counter()
        payload = _ollama_post(
            self._client,
            self._base_url,
            "/api/embeddings",
            {"model": self.model, "prompt": text},
        )
        latency = int((time.perf_counter() - started) * 1000)
        values = payload.get("embedding")
        if not isinstance(values, list) or not values:
            raise SchemaValidationError(
                "Ollama embedding response had no 'embedding' array",
                details={"payload_keys": list(payload)},
            )
        return EmbeddingResult(
            vector=[float(v) for v in values],
            usage=Usage(input_tokens=max(1, len(text.split())), latency_ms=latency),
            model=self.model,
        )

    def embed_many(self, texts: Sequence[str]) -> List[EmbeddingResult]:
        return [self.embed(t) for t in texts]


def _ollama_post(
    client: httpx.Client, base_url: str, path: str, body: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        response = client.post(f"{base_url}{path}", json=body)
    except httpx.RequestError as exc:
        raise ProviderError(
            f"Ollama is not reachable at {base_url} ({exc}); is `ollama serve` "
            "running?",
            retryable=True,
        ) from exc
    if response.status_code >= 400:
        raise ProviderError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}",
            retryable=response.status_code >= 500,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError("Ollama returned a non-JSON body", retryable=True) from exc
