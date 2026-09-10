"""Corpus ingestion: filesystem → ``images`` rows.

Idempotent by content hash (shared requirement #5): re-running ingestion after
adding two photos inserts two rows, not fifty-two.
"""

import hashlib
import logging
import os
from typing import Dict, List

from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.logging import safe_extra
from app.db import models
from app.repositories.images import ImageRepository

logger = logging.getLogger("app.ingestion")

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def sha256_of(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_directory(
    session: Session, tenant_id: str, images_dir: str
) -> Dict[str, object]:
    """Register every supported image under ``images_dir``.

    Returns a summary rather than the rows: the caller is an endpoint or a
    seed script, and both only need the counts.
    """
    if not os.path.isdir(images_dir):
        raise ValidationError(
            f"images directory {images_dir!r} does not exist — run "
            "`python -m scripts.fetch_corpus` (or `make_placeholder_corpus`) first",
            details={"images_dir": images_dir},
        )

    repo = ImageRepository(session, tenant_id)
    created: List[str] = []
    skipped: List[str] = []
    renamed: List[str] = []

    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(SUPPORTED_SUFFIXES):
            continue
        path = os.path.join(images_dir, filename)
        if not os.path.isfile(path):
            continue

        sha = sha256_of(path)
        by_sha = repo.get_by_sha(sha)
        if by_sha is not None:
            # Same bytes already ingested. If the file was renamed, follow it
            # so the path on disk stays correct without creating a duplicate.
            if by_sha.filename != filename:
                by_sha.filename = filename
                by_sha.path = path
                renamed.append(filename)
            else:
                skipped.append(filename)
            continue

        if repo.get_by_filename(filename) is not None:
            # Same name, different bytes — the image was replaced on disk.
            existing = repo.get_by_filename(filename)
            existing.sha256 = sha
            existing.path = path
            existing.bytes = os.path.getsize(path)
            existing.status = models.IMAGE_PENDING  # must be re-tagged
            renamed.append(filename)
            continue

        repo.add(
            models.Image(
                filename=filename,
                path=path,
                sha256=sha,
                bytes=os.path.getsize(path),
                status=models.IMAGE_PENDING,
            )
        )
        created.append(filename)

    session.flush()
    summary = {
        "images_dir": images_dir,
        "created": len(created),
        "already_present": len(skipped),
        "updated": len(renamed),
        "total": repo.count(),
    }
    logger.info("corpus_ingested", extra=safe_extra(summary))
    return summary
