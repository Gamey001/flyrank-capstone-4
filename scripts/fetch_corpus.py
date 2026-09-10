"""Download the real image corpus from Openverse (free, no key, no card).

The corpus is not committed — a few MB of photos does not belong in a git
repository — so this script reproduces it from ``data/corpus/manifest.json``.
Only permissively licensed results are accepted (CC0, Public Domain Mark,
CC BY), and every download's attribution is written to
``data/corpus/attribution.json`` so the licence trail is in the repo even
though the bytes are not.

    python -m scripts.fetch_corpus              # fetch everything missing
    python -m scripts.fetch_corpus --limit 5    # try a handful first

No network? Use ``python -m scripts.make_placeholder_corpus`` instead: the
pipeline runs end to end on generated tiles.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import httpx

API = "https://api.openverse.org/v1/images/"
USER_AGENT = "flyrank-capstone-image-relevance/1.0 (educational capstone)"
ALLOWED_LICENSES = ("cc0", "pdm", "by")

DEFAULT_MANIFEST = "data/corpus/manifest.json"
DEFAULT_OUT = "data/images"
ATTRIBUTION_PATH = "data/corpus/attribution.json"


def search(client: httpx.Client, query: str, page: int = 1) -> List[dict]:
    response = client.get(
        API,
        params={
            "q": query,
            "license": ",".join(ALLOWED_LICENSES),
            "extension": "jpg",
            "page_size": 8,
            "page": page,
            "mature": "false",
        },
    )
    response.raise_for_status()
    return response.json().get("results", [])


def download(client: httpx.Client, url: str, dest: str) -> int:
    response = client.get(url, follow_redirects=True, timeout=45.0)
    response.raise_for_status()
    content = response.content
    if len(content) < 2048:
        raise ValueError(f"suspiciously small image ({len(content)} bytes)")
    with open(dest, "wb") as fh:
        fh.write(content)
    return len(content)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all entries")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.4,
                        help="Pause between requests; be polite to a free API.")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = manifest["images"]
    if args.limit:
        entries = entries[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    attribution: Dict[str, dict] = _load_attribution()
    used_ids = {v.get("openverse_id") for v in attribution.values()}

    fetched = skipped = failed = 0
    with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for entry in entries:
            slug = entry["slug"]
            dest = os.path.join(args.out, f"{slug}.jpg")
            if os.path.exists(dest) and not args.force:
                skipped += 1
                continue

            result = _pick_result(client, entry, used_ids, args.sleep)
            if result is None:
                print(f"  ! {slug}: no permissively licensed result found")
                failed += 1
                continue

            try:
                size = download(client, result["url"], dest)
            except Exception as exc:
                print(f"  ! {slug}: download failed ({exc})")
                failed += 1
                continue

            # A real photo supersedes any offline placeholder for this slug.
            # Both files would otherwise ingest as two separate images.
            _remove_other_formats(args.out, slug, keep=os.path.basename(dest))

            used_ids.add(result["id"])
            attribution[slug] = {
                "openverse_id": result["id"],
                "title": result.get("title"),
                "creator": result.get("creator"),
                "license": f"{result.get('license')}-{result.get('license_version')}",
                "license_url": result.get("license_url"),
                "source_url": result.get("foreign_landing_url"),
                "attribution": result.get("attribution"),
                "bytes": size,
            }
            fetched += 1
            print(f"  + {slug}: {result.get('title')!r} ({result.get('license')})")
            time.sleep(args.sleep)

    _save_attribution(attribution)
    print(
        f"\ncorpus: {fetched} fetched, {skipped} already present, {failed} failed"
    )
    print(f"attribution written to {ATTRIBUTION_PATH}")
    if failed:
        print(
            "Tip: re-run to retry, or fill the gaps with "
            "`python -m scripts.make_placeholder_corpus`."
        )
    return 0


def _pick_result(
    client: httpx.Client, entry: dict, used_ids: set, sleep: float
) -> Optional[dict]:
    """First unused, permissively licensed result for this entry's query."""
    for page in (1, 2):
        try:
            results = search(client, entry["search_query"], page=page)
        except Exception as exc:
            print(f"  ! {entry['slug']}: search failed ({exc})")
            return None
        for result in results:
            if result.get("id") in used_ids:
                continue
            if result.get("license") not in ALLOWED_LICENSES:
                continue
            if not result.get("url"):
                continue
            return result
        time.sleep(sleep)
    return None


def _remove_other_formats(out_dir: str, slug: str, keep: str) -> None:
    """Drop any other file sharing this slug (e.g. a generated placeholder)."""
    for name in os.listdir(out_dir):
        if name != keep and os.path.splitext(name)[0] == slug:
            os.remove(os.path.join(out_dir, name))


def _load_attribution() -> Dict[str, dict]:
    if not os.path.exists(ATTRIBUTION_PATH):
        return {}
    with open(ATTRIBUTION_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh).get("images", {})


def _save_attribution(images: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(ATTRIBUTION_PATH), exist_ok=True)
    with open(ATTRIBUTION_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": "Openverse (https://openverse.org)",
                "license_policy": list(ALLOWED_LICENSES),
                "note": (
                    "Every image below is reproduced under the licence named "
                    "in its entry. Re-run scripts/fetch_corpus.py to rebuild."
                ),
                "images": images,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
