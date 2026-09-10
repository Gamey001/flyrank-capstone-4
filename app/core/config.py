"""Application settings.

Every knob the system has is here, loaded from the environment (see
``.env.example``). Nothing in this module ever reads a hard-coded secret.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VisionProvider = Literal["stub", "gemini", "ollama"]
EmbeddingProvider = Literal["stub", "gemini", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- app -------------------------------------------------------------
    app_env: Literal["local", "test", "docker", "prod"] = "local"
    log_level: str = "INFO"
    api_prefix: str = "/v1"

    # --- persistence -----------------------------------------------------
    # Postgres is the documented path (docker compose); SQLite keeps the test
    # suite and a keyless local demo runnable with no services at all.
    database_url: str = "sqlite+pysqlite:///./flyrank.sqlite3"
    default_tenant: str = "demo"

    # --- corpus ----------------------------------------------------------
    images_dir: str = "data/images"
    corpus_manifest: str = "data/corpus/manifest.json"

    # --- AI providers ----------------------------------------------------
    vision_provider: VisionProvider = "stub"
    embedding_provider: EmbeddingProvider = "stub"

    gemini_api_key: Optional[str] = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_vision_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "llava"
    ollama_embedding_model: str = "all-minilm"

    ai_request_timeout_s: float = 60.0

    # --- mismatch guard thresholds --------------------------------------
    # Defaults are the values tuned against the labeled eval set; see
    # ``scripts/tune_thresholds.py`` and the numbers reported in README.md.
    similarity_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    min_vision_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.03, ge=0.0, le=1.0)

    # --- background jobs -------------------------------------------------
    worker_enabled: bool = True
    worker_poll_interval_s: float = 0.5
    job_max_attempts: int = 3
    job_backoff_base_s: float = 0.5
    job_backoff_max_s: float = 30.0
    alert_webhook_url: Optional[str] = None

    # --- cost control ----------------------------------------------------
    ai_daily_budget_usd: float = Field(default=1.0, ge=0.0)
    # Free tier still costs $0; we price calls at the published paid rate so
    # the cost log is meaningful and the budget guard is actually exercised.
    vision_cost_per_call_usd: float = 0.000_2
    embedding_cost_per_1k_tokens_usd: float = 0.000_025

    @field_validator("low_confidence_threshold")
    @classmethod
    def _flag_above_reject(cls, v: float, info) -> float:
        floor = info.data.get("min_vision_confidence", 0.0)
        if v < floor:
            raise ValueError(
                "low_confidence_threshold must be >= min_vision_confidence"
            )
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
