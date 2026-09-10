"""A small, explicit concept lexicon.

Two consumers:

1. The **mismatch guard** uses it to tell "the post wants a *fox*, the model
   detected a *wolf*" apart from "the post wants a fox and the image is a
   fox photographed in snow". Cosine similarity alone cannot make that call —
   fox and wolf captions are genuinely close in vector space, which is exactly
   why a separate tag-level check exists.

2. The **offline embedding provider** (``providers/stub.py``) projects text
   onto these concepts, which is what makes "red fox" ≈ "Vulpes vulpes" work
   with no API key and no network.

It is deliberately hand-written and small: the corpus is ~50 images across a
handful of subjects, and a lexicon a reviewer can read in one screen is worth
more here than an opaque model.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

CATEGORY_ANIMAL = "animal"
CATEGORY_LANDSCAPE = "landscape"
CATEGORY_FOOD = "food"
CATEGORY_ARCHITECTURE = "architecture"


@dataclass(frozen=True)
class Concept:
    key: str
    category: str
    #: surface forms that mean this concept, lowercase
    synonyms: FrozenSet[str]
    #: soft associations — they add similarity but never satisfy a subject check
    related: FrozenSet[str] = field(default_factory=frozenset)


def _c(
    key: str, category: str, synonyms: List[str], related: List[str] = ()
) -> Concept:
    return Concept(key, category, frozenset(synonyms), frozenset(related))


#: Subject concepts — the thing an image is *of*.
SUBJECTS: Dict[str, Concept] = {
    c.key: c
    for c in [
        _c("red fox", CATEGORY_ANIMAL,
           ["red fox", "fox", "foxes", "vulpes", "vulpes vulpes", "wild fox",
            "fox cub", "fox kit", "renard"],
           ["wild", "orange fur", "bushy tail", "forest"]),
        _c("gray wolf", CATEGORY_ANIMAL,
           ["gray wolf", "grey wolf", "wolf", "wolves", "canis lupus",
            "timber wolf", "wolf pack"],
           ["wild", "pack", "howling", "forest", "snow"]),
        _c("dog", CATEGORY_ANIMAL,
           ["dog", "dogs", "domestic dog", "puppy", "puppies", "canis familiaris",
            "golden retriever", "labrador", "husky", "pet dog"],
           ["pet", "domestic", "collar", "park"]),
        _c("brown bear", CATEGORY_ANIMAL,
           ["brown bear", "bear", "bears", "grizzly", "grizzly bear", "ursus",
            "ursus arctos"],
           ["wild", "river", "salmon", "forest"]),
        _c("deer", CATEGORY_ANIMAL,
           ["deer", "roe deer", "red deer", "stag", "doe", "fawn", "cervidae",
            "buck", "antlers"],
           ["wild", "meadow", "forest", "antlers"]),
        _c("forest", CATEGORY_LANDSCAPE,
           ["forest", "woods", "woodland", "pine forest", "trees", "grove"],
           ["green", "trees", "mist", "nature"]),
        _c("mountain", CATEGORY_LANDSCAPE,
           ["mountain", "mountains", "peak", "summit", "alps", "mountain range",
            "ridge"],
           ["snow", "rock", "altitude", "nature"]),
        _c("beach", CATEGORY_LANDSCAPE,
           ["beach", "shore", "coast", "coastline", "seaside", "sandy beach"],
           ["sand", "ocean", "waves", "nature"]),
        _c("pizza", CATEGORY_FOOD,
           ["pizza", "margherita", "pizza slice", "neapolitan pizza"],
           ["cheese", "tomato", "baked", "meal"]),
        _c("coffee", CATEGORY_FOOD,
           ["coffee", "espresso", "latte", "cappuccino", "coffee cup",
            "flat white"],
           ["cup", "cafe", "drink", "beans"]),
        _c("bridge", CATEGORY_ARCHITECTURE,
           ["bridge", "suspension bridge", "viaduct", "footbridge"],
           ["steel", "river", "span", "city"]),
        _c("cathedral", CATEGORY_ARCHITECTURE,
           ["cathedral", "church", "basilica", "gothic cathedral", "chapel"],
           ["gothic", "stone", "arches", "city"]),
    ]
}

#: Modifier concepts — setting/appearance. They refine ranking between two
#: images of the same subject ("fox in snow" vs "fox in forest").
MODIFIERS: Dict[str, Concept] = {
    c.key: c
    for c in [
        _c("snow", "modifier", ["snow", "snowy", "winter", "frost", "snowfall"]),
        _c("forest setting", "modifier",
           ["forest", "woods", "woodland", "trees", "undergrowth"]),
        _c("meadow", "modifier", ["meadow", "field", "grassland", "grass", "pasture"]),
        # "dark" is deliberately absent: a dark roast coffee is not a night shot.
        _c("night", "modifier",
           ["night", "nocturnal", "moonlight", "dusk", "midnight", "starlit"]),
        _c("close-up", "modifier",
           ["close-up", "closeup", "portrait", "macro", "detail shot"]),
        _c("water", "modifier", ["water", "river", "lake", "stream", "ocean", "sea"]),
        _c("urban", "modifier", ["urban", "city", "street", "downtown", "town"]),
        _c("mist", "modifier", ["mist", "misty", "fog", "foggy", "haze"]),
        _c("pet context", "modifier",
           ["pet", "domestic", "owner", "leash", "collar", "backyard"]),
        _c("wild context", "modifier",
           ["wild", "wildlife", "wilderness", "untamed", "predator"]),
    ]
}

ALL_CONCEPTS: Dict[str, Concept] = {**SUBJECTS, **MODIFIERS}

#: Subjects that look alike and are routinely confused. A candidate whose
#: subject sits in the same group as — but differs from — the subject the post
#: asks for is the classic wrong match, and the guard rejects it outright.
CONFUSABLE_GROUPS: List[FrozenSet[str]] = [
    frozenset({"red fox", "gray wolf", "dog"}),   # canids
    frozenset({"brown bear", "gray wolf"}),       # large wild mammals
    frozenset({"deer", "brown bear"}),            # forest megafauna
    frozenset({"forest", "mountain", "beach"}),   # landscape types
    frozenset({"bridge", "cathedral"}),           # structures
]

#: Longest synonym first so "red fox" wins over "fox", "gray wolf" over "wolf".
_SYNONYM_INDEX = sorted(
    (
        (syn, concept.key)
        for concept in ALL_CONCEPTS.values()
        for syn in concept.synonyms
    ),
    key=lambda pair: (-len(pair[0]), pair[0]),
)
_RELATED_INDEX = sorted(
    (
        (term, concept.key)
        for concept in ALL_CONCEPTS.values()
        for term in concept.related
    ),
    key=lambda pair: (-len(pair[0]), pair[0]),
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def normalize(text: str) -> str:
    """Lowercase and collapse to space-separated word tokens."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _match_terms(haystack: str, index) -> Set[str]:
    padded = f" {haystack} "
    found: Set[str] = set()
    for term, key in index:
        if f" {term} " in padded:
            found.add(key)
    return found


def extract_subjects(text: str) -> Set[str]:
    """Subject concepts explicitly named in ``text``."""
    return {
        key
        for key in _match_terms(normalize(text), _SYNONYM_INDEX)
        if key in SUBJECTS
    }


def extract_modifiers(text: str) -> Set[str]:
    """Setting/appearance concepts named in ``text``."""
    return {
        key
        for key in _match_terms(normalize(text), _SYNONYM_INDEX)
        if key in MODIFIERS
    }


def extract_related(text: str) -> Set[str]:
    """Soft associations — weighted lower and never used for subject checks."""
    return _match_terms(normalize(text), _RELATED_INDEX)


def category_of(subject_key: str) -> Optional[str]:
    concept = SUBJECTS.get(subject_key)
    return concept.category if concept else None


def canonical_subject(text: str) -> Optional[str]:
    """Best single subject for a free-text subject string (e.g. the model's).

    Prefers the longest matching surface form, so ``"a red fox"`` resolves to
    ``red fox`` rather than to the shorter ``fox`` synonym of the same concept.
    """
    padded = f" {normalize(text)} "
    for term, key in _SYNONYM_INDEX:
        if f" {term} " in padded and key in SUBJECTS:
            return key
    return None


def are_confusable(a: str, b: str) -> bool:
    """True when two *different* subjects are a known look-alike pair."""
    if a == b:
        return False
    return any(a in group and b in group for group in CONFUSABLE_GROUPS)
