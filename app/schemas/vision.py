"""The contract every vision model must satisfy.

Nothing downstream ever touches a raw model response: it is parsed through
``ImageTags`` first, and a response that does not fit is a *provider error*,
not data. That is the "never trust invalid model output" rule made mechanical.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Handed to providers that support native structured output (Gemini's
#: ``responseSchema``, Ollama's ``format``) so the model is constrained at the
#: source as well as validated on arrival.
IMAGE_TAGS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "minLength": 2, "maxLength": 200},
        "category": {
            "type": "string",
            "enum": ["animal", "landscape", "food", "architecture", "other"],
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string", "minLength": 2, "maxLength": 60},
            "maxItems": 12,
        },
        "caption": {"type": "string", "minLength": 5, "maxLength": 400},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["subject", "category", "attributes", "caption", "confidence"],
}

ALLOWED_CATEGORIES = tuple(IMAGE_TAGS_JSON_SCHEMA["properties"]["category"]["enum"])


class ImageTags(BaseModel):
    """Validated vision output for a single image."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    subject: str = Field(min_length=2, max_length=200)
    category: str
    attributes: List[str] = Field(default_factory=list, max_length=12)
    caption: str = Field(min_length=5, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        value = v.strip().lower()
        if value not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"category must be one of {ALLOWED_CATEGORIES}, got {v!r}"
            )
        return value

    @field_validator("subject")
    @classmethod
    def _normalize_subject(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("attributes")
    @classmethod
    def _clean_attributes(cls, v: List[str]) -> List[str]:
        seen, out = set(), []
        for item in v:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def embedding_text(self) -> str:
        """The text that represents this image in the shared semantic space."""
        return " ".join(
            [self.caption, self.subject, self.category, *self.attributes]
        ).strip()
