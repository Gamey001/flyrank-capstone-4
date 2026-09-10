"""Structured (JSON-lines) logging.

One line per event keeps job/cost/guard traces greppable in EVIDENCE.md.
"""

import json
import logging
import sys
from typing import Any

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def safe_extra(fields: dict) -> dict:
    """Make an arbitrary dict safe to pass as ``extra=``.

    ``logging.Logger.makeRecord`` raises if a key collides with a LogRecord
    attribute — ``created``, ``module``, ``name``, ``filename`` and friends.
    Summary dicts built elsewhere in the app naturally contain words like
    "created", so any such key is suffixed rather than allowed to blow up a
    request at INFO level and pass silently at WARNING.
    """
    return {
        (f"{key}_" if key in _RESERVED else key): value
        for key, value in fields.items()
    }


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn brings its own noisy handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False
