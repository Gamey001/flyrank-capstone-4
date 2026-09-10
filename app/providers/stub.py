"""Deterministic offline providers.

Why these exist
---------------
A reviewer must be able to clone the repo and see the whole pipeline — batch
tagging, ranking, the guard refusing the wolf, the eval number — with no API
key, no network and no GPU. These providers make that possible, and they make
the test suite deterministic.

They are **fixtures, not models**. Honest about it:

* ``FixtureVisionProvider`` reads the ground truth from the corpus manifest
  (falling back to the filename slug) and returns it in the same validated
  shape a real model would, including a deliberately low confidence on the
  images the manifest marks ``uncertain`` — that is what PROBE 1's
  "at least one low-confidence image is flagged" exercises.
* ``LexiconEmbeddingProvider`` projects text onto the concept lexicon in
  ``services/concepts.py``. It is a real vector space with real cosine
  behaviour — "red fox" and "Vulpes vulpes" land on the same axis — but its
  vocabulary is the lexicon, not the language.

Set ``VISION_PROVIDER=gemini`` / ``EMBEDDING_PROVIDER=gemini`` (or ``ollama``)
to run the identical pipeline against an actual model.
"""

import json
import math
import os
import re
import time
from typing import Dict, List, Optional, Sequence

from app.core.errors import ProviderError
from app.providers.base import EmbeddingResult, Usage, VisionResult
from app.schemas.vision import ImageTags
from app.services import concepts

_SLUG_RE = re.compile(r"[^a-z]+")


class FixtureVisionProvider:
    """Vision output replayed from the corpus manifest / filename."""

    name = "fixture"
    model = "fixture-vision-v1"

    def __init__(self, manifest_path: Optional[str] = None):
        self._entries: Dict[str, dict] = {}
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            # Keyed on slug: the same entry serves red-fox-snow-01.jpg and
            # the .png the offline placeholder generator writes.
            self._entries = {
                e.get("slug") or os.path.splitext(e["filename"])[0]: e
                for e in manifest.get("images", [])
            }

    def describe_image(
        self, image_path: str, *, hint: Optional[str] = None
    ) -> VisionResult:
        started = time.perf_counter()
        filename = os.path.basename(image_path)
        slug = os.path.splitext(filename)[0]
        entry = self._entries.get(slug)

        if entry is None:
            entry = self._from_filename(slug)
        if entry is None:
            raise ProviderError(
                f"fixture provider has no ground truth for {filename!r}; add it "
                "to data/corpus/manifest.json or use a real vision provider",
                retryable=False,
            )

        raw = {
            "subject": entry["subject"],
            "category": entry["category"],
            "attributes": list(entry.get("attributes", [])),
            "caption": entry.get("caption") or self._caption(entry),
            "confidence": float(entry.get("confidence", 0.93)),
        }
        # Parsed through the same validator a live model's output goes through.
        tags = ImageTags.model_validate(raw)
        latency = int((time.perf_counter() - started) * 1000)
        return VisionResult(
            tags=tags,
            raw=raw,
            usage=Usage(
                input_tokens=280,  # a ~512px image tile plus the prompt
                output_tokens=60,
                latency_ms=latency,
                meta={"source": "fixture", "filename": filename},
            ),
            model=self.model,
        )

    @staticmethod
    def _from_filename(slug: str) -> Optional[dict]:
        """`red-fox-snow-01` → subject "red fox", attribute "snow"."""
        words = " ".join(w for w in _SLUG_RE.split(slug.lower()) if w)
        subject = concepts.canonical_subject(words)
        if not subject:
            return None
        return {
            "subject": subject,
            "category": concepts.category_of(subject) or "other",
            "attributes": sorted(concepts.extract_modifiers(words)),
            "confidence": 0.9,
        }

    @staticmethod
    def _caption(entry: dict) -> str:
        attrs = ", ".join(entry.get("attributes", []))
        base = f"A photo of a {entry['subject']}"
        return f"{base} ({attrs})" if attrs else base


class LexiconEmbeddingProvider:
    """Concept-space embeddings: one dimension per concept in the lexicon."""

    name = "fixture"
    model = "lexicon-concept-v1"

    #: Subject terms dominate the vector; modifiers refine within a subject;
    #: related terms only nudge. Without this weighting "fox in snow" and
    #: "wolf in snow" would rank equally for a snow-fox post.
    W_SUBJECT = 1.0
    W_MODIFIER = 0.45
    W_RELATED = 0.15

    def __init__(self) -> None:
        self._axes: List[str] = sorted(concepts.ALL_CONCEPTS)
        self._index = {key: i for i, key in enumerate(self._axes)}
        self.dim = len(self._axes) + 1  # +1 lexical-fallback axis

    def embed(self, text: str) -> EmbeddingResult:
        started = time.perf_counter()
        vec = [0.0] * self.dim

        for key in concepts.extract_subjects(text):
            vec[self._index[key]] += self.W_SUBJECT
        for key in concepts.extract_modifiers(text):
            vec[self._index[key]] += self.W_MODIFIER
        for key in concepts.extract_related(text):
            vec[self._index[key]] += self.W_RELATED

        if not any(vec):
            # Out-of-lexicon text still gets a stable, non-zero vector so it can
            # be compared — it simply will not be close to anything meaningful.
            vec[-1] = 1.0

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        latency = int((time.perf_counter() - started) * 1000)
        return EmbeddingResult(
            vector=vec,
            usage=Usage(
                input_tokens=max(1, len(text.split())),
                latency_ms=latency,
                meta={"source": "fixture"},
            ),
            model=self.model,
        )

    def embed_many(self, texts: Sequence[str]) -> List[EmbeddingResult]:
        return [self.embed(t) for t in texts]
