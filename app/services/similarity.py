"""Vector math.

Pure Python: at ~50 images a full scan is microseconds and it keeps numpy off
the dependency list. See the README scaling note for where pgvector takes over.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, clamped to [-1, 1]."""
    if len(a) != len(b):
        raise ValueError(
            f"vector dimension mismatch: {len(a)} vs {len(b)} — the image and "
            "post embeddings were produced by different models"
        )
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))


def rank_by_similarity(
    query: Sequence[float],
    candidates: Dict[str, List[float]],
    tiebreak: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, float]]:
    """Every candidate scored and sorted, best first.

    Exact ties are real — two images can carry the same concepts — so they are
    broken on a caller-supplied stable key (the filename), never on the row id.
    Row ids are random UUIDs, and sorting on them would make the eval number
    change between seeds of the same data.
    """
    keys = tiebreak or {}
    scored = [
        (owner_id, cosine_similarity(query, vector))
        for owner_id, vector in candidates.items()
    ]
    scored.sort(key=lambda pair: (-pair[1], keys.get(pair[0], pair[0])))
    return scored
