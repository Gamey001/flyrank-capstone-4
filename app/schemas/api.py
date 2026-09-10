"""Request/response models — the HTTP boundary contract.

Everything entering the system is parsed here first, so a malformed request is
a 422 with a field-level explanation and never reaches a service (shared
requirement #2).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorResponse(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


# --- images ------------------------------------------------------------------
class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    subject: Optional[str] = None
    category: Optional[str] = None
    caption: Optional[str] = None
    confidence: Optional[float] = None
    attributes: List[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: Optional[str] = None
    vision_model: Optional[str] = None
    last_error: Optional[str] = None
    tagged_at: Optional[datetime] = None

    @field_validator("attributes", mode="before")
    @classmethod
    def _null_to_list(cls, v):
        return v or []


class IngestRequest(BaseModel):
    images_dir: Optional[str] = Field(
        default=None,
        description="Override the configured corpus directory.",
        max_length=1024,
    )


class IngestResponse(BaseModel):
    images_dir: str
    created: int
    already_present: int
    updated: int
    total: int


# --- posts -------------------------------------------------------------------
class PostCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    slug: str = Field(min_length=1, max_length=200, pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=3, max_length=300)
    body: str = Field(min_length=10, max_length=20_000)


class PostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    slug: str
    title: str
    body: str


# --- suggestions / guard -----------------------------------------------------
class ReasonOut(BaseModel):
    code: str
    message: str


class SuggestionOut(BaseModel):
    suggestion_id: Optional[str] = None
    image_id: str
    filename: str
    rank: int
    similarity: float
    accepted: bool
    subject: Optional[str] = None
    category: Optional[str] = None
    caption: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: bool = False
    reasons: List[ReasonOut] = Field(default_factory=list)


class NoMatchOut(BaseModel):
    message: str
    wanted_subjects: List[str] = Field(default_factory=list)
    best_similarity: float = 0.0
    similarity_threshold: float
    rejection_counts: Dict[str, int] = Field(default_factory=dict)
    candidates_considered: int = 0


class MatchResponse(BaseModel):
    post_id: str
    post_slug: str
    post_title: str
    embedding_model: str
    has_confident_match: bool
    #: the accepted pick, or null when the guard refused everything
    best_match: Optional[SuggestionOut] = None
    suggestions: List[SuggestionOut] = Field(default_factory=list)
    rejected: List[SuggestionOut] = Field(default_factory=list)
    no_match: Optional[NoMatchOut] = None


class GuardCheckResponse(BaseModel):
    post_id: str
    post_title: str
    image_id: str
    filename: str
    similarity: float
    accepted: bool
    reasons: List[ReasonOut]
    signals: Dict[str, Any]


class ExplainResponse(BaseModel):
    suggestion_id: str
    post_id: str
    post_title: str
    image_id: str
    filename: str
    rank: int
    similarity: float
    verdict: str
    reasons: List[ReasonOut]
    signals: Dict[str, Any]
    image_embedding_text: Optional[str] = None
    post_embedding_text: Optional[str] = None
    review: Optional["ReviewOut"] = None


# --- review ------------------------------------------------------------------
class ReviewCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    suggestion_id: str = Field(min_length=1, max_length=32)
    decision: str = Field(description="approved | rejected")
    reviewer: str = Field(min_length=1, max_length=120)
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        return value


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    suggestion_id: str
    decision: str
    reviewer: str
    note: Optional[str] = None
    idempotency_key: str
    created_at: datetime


# --- jobs --------------------------------------------------------------------
class JobCreate(BaseModel):
    kind: str = Field(
        description=(
            "vision_tagging | embed_images | embed_posts | match_posts | "
            "full_pipeline"
        )
    )
    dedupe_key: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Submitting the same key twice returns the first job instead of "
            "queueing a duplicate."
        ),
    )
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        from app.services.jobs import JOB_KINDS

        value = v.strip().lower()
        if value not in JOB_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(JOB_KINDS)}")
        return value


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    dedupe_key: str
    attempts: int
    max_attempts: int
    total_items: int
    processed_items: int
    failed_items: int
    result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# --- costs -------------------------------------------------------------------
class AiCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    operation: str
    provider: str
    model: str
    subject_ref: str
    job_id: Optional[str] = None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    status: str
    created_at: datetime


class CostSummaryOut(BaseModel):
    total_calls: int
    total_cost_usd: float
    failed_calls: int
    daily_budget_usd: float
    spent_last_24h_usd: float
    budget_remaining_usd: float
    by_operation: List[Dict[str, Any]]


# --- health ------------------------------------------------------------------
class HealthOut(BaseModel):
    status: str
    version: str
    vision_provider: str
    embedding_provider: str
    database: str


ExplainResponse.model_rebuild()
