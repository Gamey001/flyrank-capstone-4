"""Provider selection — the only place that maps config to an implementation."""

from typing import Tuple

from app.core.config import Settings
from app.providers.base import EmbeddingProvider, VisionProvider
from app.providers.stub import FixtureVisionProvider, LexiconEmbeddingProvider


def build_vision_provider(settings: Settings) -> VisionProvider:
    if settings.vision_provider == "gemini":
        from app.providers.gemini import GeminiVisionProvider

        return GeminiVisionProvider(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_vision_model,
            base_url=settings.gemini_base_url,
            timeout_s=settings.ai_request_timeout_s,
        )
    if settings.vision_provider == "ollama":
        from app.providers.ollama import OllamaVisionProvider

        return OllamaVisionProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_vision_model,
            timeout_s=settings.ai_request_timeout_s,
        )
    return FixtureVisionProvider(manifest_path=settings.corpus_manifest)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "gemini":
        from app.providers.gemini import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_embedding_model,
            base_url=settings.gemini_base_url,
            timeout_s=settings.ai_request_timeout_s,
        )
    if settings.embedding_provider == "ollama":
        from app.providers.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            timeout_s=settings.ai_request_timeout_s,
        )
    return LexiconEmbeddingProvider()


def build_providers(settings: Settings) -> Tuple[VisionProvider, EmbeddingProvider]:
    return build_vision_provider(settings), build_embedding_provider(settings)
