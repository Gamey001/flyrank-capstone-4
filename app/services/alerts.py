"""Failure alerting for background work.

A job that exhausts its retries must be *loud*: a CRITICAL structured log line
that a log-based alert can match on, plus an optional webhook. Silent job death
is the classic way a batch pipeline rots.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from app.core.logging import safe_extra

logger = logging.getLogger("app.alerts")

#: Grep target for log-based alerting, e.g. `grep JOB_FAILURE_ALERT`.
ALERT_MARKER = "JOB_FAILURE_ALERT"


def send_job_failure_alert(
    *,
    job_id: str,
    tenant_id: str,
    kind: str,
    attempts: int,
    error: str,
    webhook_url: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "alert": ALERT_MARKER,
        "job_id": job_id,
        "tenant_id": tenant_id,
        "kind": kind,
        "attempts": attempts,
        "error": error[:1000],
        "context": context or {},
    }
    logger.critical(ALERT_MARKER, extra=safe_extra(payload))

    if not webhook_url:
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(webhook_url, json=payload)
    except Exception as exc:  # never let alerting break the worker
        logger.error(
            "alert webhook failed",
            extra=safe_extra({"error": str(exc), "job_id": job_id}),
        )
