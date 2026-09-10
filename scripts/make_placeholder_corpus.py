"""Generate a deterministic offline stand-in corpus.

Why: the real corpus is downloaded (``scripts/fetch_corpus.py``) rather than
committed, because a few MB of photos does not belong in a git repository. But
a reviewer with no network must still be able to run the whole pipeline. This
writes one small, valid, deterministic JPEG per manifest entry so ingestion,
hashing, batch jobs, ranking and the guard all have real files to work on.

These are **not photographs** — they are flat colour tiles. They exercise the
plumbing, not a vision model. Pair them with ``VISION_PROVIDER=stub``; a real
vision provider pointed at these would (correctly) describe coloured squares.

No third-party imaging library: a baseline JPEG is assembled by hand so the
script works on a bare Python install.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from typing import List, Tuple

DEFAULT_MANIFEST = "data/corpus/manifest.json"
DEFAULT_OUT = "data/images"


def _png_bytes(width: int, height: int, rgb: Tuple[int, int, int]) -> bytes:
    """A minimal valid PNG of a solid colour."""
    raw = b"".join(
        b"\x00" + bytes(rgb) * width for _ in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _colour_for(name: str) -> Tuple[int, int, int]:
    """Stable per-image colour so files differ (and hash differently)."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    # Keep it mid-tone so the tiles are visually distinguishable.
    return tuple(60 + (b % 160) for b in digest[:3])


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist (e.g. real downloads).",
    )
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    existing_slugs = {
        os.path.splitext(name)[0] for name in os.listdir(args.out)
    }
    written = skipped = 0
    for entry in manifest["images"]:
        # PNG bytes under a .jpg name would be dishonest; write real .png files
        # and let ingestion pick them up by suffix.
        slug = entry["slug"]
        filename = f"{slug}.png"
        path = os.path.join(args.out, filename)
        # Never shadow a real download: two files sharing a slug would ingest
        # as two separate images.
        if slug in existing_slugs and not args.force:
            skipped += 1
            continue
        if os.path.exists(path) and not args.force:
            skipped += 1
            continue
        with open(path, "wb") as fh:
            fh.write(_png_bytes(args.size, args.size, _colour_for(filename)))
        written += 1

    print(
        f"placeholder corpus: {written} written, {skipped} already present "
        f"in {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
