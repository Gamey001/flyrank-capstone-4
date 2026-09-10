"""Domain errors and their HTTP mapping.

Rule 2 of the shared requirements: bad input produces a clean 4xx with a
machine-readable body, never a 500.
"""

from typing import Any, Optional


class AppError(Exception):
    """Base class for errors that are safe to show a client."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class BudgetExceededError(AppError):
    """The daily AI spend cap was hit — refuse rather than overspend."""

    status_code = 429
    code = "ai_budget_exceeded"


class ProviderError(AppError):
    """An upstream AI provider failed. Retryable unless stated otherwise."""

    status_code = 502
    code = "provider_error"

    def __init__(self, message: str, *, retryable: bool = True, details=None):
        super().__init__(message, details=details)
        self.retryable = retryable


class SchemaValidationError(ProviderError):
    """The model answered, but not in the shape we demanded."""

    status_code = 502
    code = "model_schema_invalid"
