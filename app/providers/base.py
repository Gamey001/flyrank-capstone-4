"""Provider interfaces.

Both AI capabilities are behind a Protocol so the pipeline is identical
whether it runs on Gemini Flash, a local Ollama model, or the deterministic
offline fixture provider used by tests and the keyless demo.

Providers return *usage* alongside the payload; the cost ledger is written by
the caller, so no provider can quietly skip accounting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from app.schemas.vision import ImageTags


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionResult:
    tags: ImageTags
    raw: Dict[str, Any]
    usage: Usage
    model: str


@dataclass
class EmbeddingResult:
    vector: List[float]
    usage: Usage
    model: str

    @property
    def dim(self) -> int:
        return len(self.vector)


class VisionProvider(Protocol):
    name: str
    model: str

    def describe_image(
        self, image_path: str, *, hint: Optional[str] = None
    ) -> VisionResult:
        """Return schema-valid tags, or raise ``ProviderError``."""


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text into the shared semantic space."""

    def embed_many(self, texts: Sequence[str]) -> List[EmbeddingResult]:
        """Embed a batch; default implementations may loop."""
