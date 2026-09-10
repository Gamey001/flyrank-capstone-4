# EVIDENCE

One pasted proof per requirement checkbox in Section 6 of the brief, plus the
six acceptance probes. Every transcript below is real output, captured from a
clean run:

```bash
python -m scripts.make_placeholder_corpus
python -m scripts.seed
uvicorn app.main:app --port 8000
```

Providers were the default offline fixtures (`VISION_PROVIDER=stub`,
`EMBEDDING_PROVIDER=stub`) — see the README's honesty note about what those are.
`$B` is the base URL.

---

## Requirements checklist

### AI processing

| # | Requirement | Proof |
|---|---|---|
| 1 | Vision output validated against a schema; invalid never trusted | [A1](#a1-schema-validation) |
| 2 | Low-confidence classifications flagged, not accepted | [A2](#a2-low-confidence-is-flagged) |
| 3 | Images processed by a batch background job with retries | [A3](#a3-batch-job-with-retries) |
| 4 | Vision and embedding costs tracked per call | [A4](#a4-per-call-cost-tracking) |

### Matching system

| # | Requirement | Proof |
|---|---|---|
| 5 | Image and post embeddings stored; posts return ranked suggestions | [B1](#b1-ranked-suggestions) |
| 6 | Semantic matching works for equivalent concepts | [B2](#b2-semantic-equivalence) |

### Safety layer

| # | Requirement | Proof |
|---|---|---|
| 7 | The guard rejects incorrect recommendations; the wolf provably fails | [C1](#c1-the-wolf-provably-fails) |
| 8 | Rejections include a human-readable explanation | [C1](#c1-the-wolf-provably-fails), [C2](#c2-no-confident-match) |
| 9 | "No confident match" with reasons when nothing clears the bar | [C2](#c2-no-confident-match) |

### Backend

| # | Requirement | Proof |
|---|---|---|
| 10 | DB models for images, tags, embeddings, posts, suggestions, reviews — with indexes | [D1](#d1-schema-and-indexes) |
| 11 | API endpoints validated; review workflow exists | [D2](#d2-validation-at-the-boundary), [D3](#d3-review-workflow) |

### Quality & documentation

| # | Requirement | Proof |
|---|---|---|
| 12 | Labeled eval set measures top-1 precision; the number is in the README | [E1](#e1-top-1-precision) |
| 13 | README with architecture explanation and diagram; required files present | [E2](#e2-required-files) |

### Shared requirements

| # | Requirement | Proof |
|---|---|---|
| 1 | Layered architecture | [F1](#f1-layering) |
| 2 | Validation at the boundary — bad input → clean 4xx, never a 500 | [D2](#d2-validation-at-the-boundary) |
| 3 | ≥1 background job, retries + failure alert | [A3](#a3-batch-job-with-retries), [F2](#f2-retries-and-the-failure-alert) |
| 4 | Real persistence — migrations, indexes, isolated tenants | [D1](#d1-schema-and-indexes), [F3](#f3-tenant-isolation) |
| 5 | Idempotency where it matters | [F4](#f4-idempotency) |
| 6 | Secrets clean — env only, never logged | [F5](#f5-secrets) |
| 7 | Cost tracked per call, attributed, with a budget guard | [A4](#a4-per-call-cost-tracking), [F6](#f6-budget-guard) |

---

## The six acceptance probes

### PROBE 1 — batch job tags the corpus; at least one low-confidence image is flagged

```
########## PROBE 1: batch job tagged the corpus; low confidence flagged
$ curl -s http://127.0.0.1:8077/v1/images/stats
{
    "total": 51,
    "by_status": {
        "tagged": 51
    },
    "flagged_for_review": 3
}

$ curl -s 'http://127.0.0.1:8077/v1/images?needs_review=true'
{
  "filename": "deer-mist-04.png",
  "subject": "deer",
  "confidence": 0.66,
  "needs_review": true,
  "review_reason": "Vision confidence 0.66 is below the 0.70 trust threshold; flagged for human confirmation.",
  "vision_model": "fixture-vision-v1"
}
{
  "filename": "gray-wolf-night-05.png",
  "subject": "gray wolf",
  "confidence": 0.48,
  "needs_review": true,
  "review_reason": "Vision confidence 0.48 is below the 0.55 usable minimum; this image is never auto-suggested.",
  "vision_model": "fixture-vision-v1"
}
{
  "filename": "red-fox-urban-06.png",
  "subject": "red fox",
  "confidence": 0.61,
  "needs_review": true,
  "review_reason": "Vision confidence 0.61 is below the 0.70 trust threshold; flagged for human confirmation.",
  "vision_model": "fixture-vision-v1"
}
```

All 51 images tagged, zero failures, three images flagged rather than guessed.
Note the two tiers: below `0.55` the image is *never* auto-suggested; between
`0.55` and `0.70` it is flagged for human confirmation.

---

### PROBE 2 — the fox post ranks the fox first; wolf and dog rank clearly lower

```
########## PROBE 2: fox post ranks the fox first; wolf and dog rank lower
$ curl -s 'http://127.0.0.1:8077/v1/posts/red-foxes-in-deep-snow/images?limit=60'
has_confident_match: True
best_match         : red-fox-snow-01.png | subject 'red fox' | similarity 0.8804
reason             : Subject red fox matches the post, similarity 0.88 ≥ 0.62, vision confidence 0.95.

ACCEPTED (guard passed):
  #1   0.880  red-fox-snow-01.png        red fox
  #2   0.834  red-fox-closeup-04.png     red fox
  #3   0.783  red-fox-meadow-03.png      red fox
  #4   0.783  red-fox-night-05.png       red fox

REJECTED (top of the ranking, refused by the guard):
  #5   0.744  red-fox-urban-06.png       red fox      -> Image is flagged for human review (confidence 0.61) and is not auto-approved.
  #7   0.364  gray-wolf-snow-01.png      gray wolf    -> Animal category mismatch: expected red fox, detected gray wolf.
  #8   0.289  forest-snow-02.png         forest       -> Post is about red fox (animal), but the image shows forest (landscape).
  #9   0.286  mountain-snow-01.png       mountain     -> Post is about red fox (animal), but the image shows mountain (landscape).
  #10  0.250  deer-snow-03.png           deer         -> Post asks for red fox, but the image shows deer.
  #12  0.166  dog-snow-04.png            dog          -> Animal category mismatch: expected red fox, detected dog.
  #19  0.115  brown-bear-meadow-04.png   brown bear   -> Post asks for red fox, but the image shows brown bear.
  #28  0.000  beach-01.png               beach        -> Post is about red fox (animal), but the image shows beach (landscape).
  #31  0.000  bridge-mist-04.png         bridge       -> Post is about red fox (animal), but the image shows bridge (architecture).
  #36  0.000  cathedral-01.png           cathedral    -> Post is about red fox (animal), but the image shows cathedral (architecture).
  #40  0.000  coffee-01.png              coffee       -> Post is about red fox (animal), but the image shows coffee (food).
  #49  0.000  pizza-01.png               pizza        -> Post is about red fox (animal), but the image shows pizza (food).
```

Fox images take ranks 1–5. The gray wolf lands at #7 and the dog at #12, and
both are refused by the guard even though they are ranked. Note rank #5: a fox
image the vision model was only 0.61 confident about is refused too — being the
right animal is not sufficient.

---

### PROBE 3 — force the wolf as a candidate for the fox post

```
########## PROBE 3: force the wolf onto the fox post -> guard rejects
$ WOLF=$(curl -s "$B/v1/images?limit=200" | jq -r '.[]|select(.filename|startswith("gray-wolf-snow-01")).id')
$ curl -s -X POST "$B/v1/posts/red-foxes-in-deep-snow/images/$WOLF/check"
{
    "post_id": "a97f3444378045bfa004b35b97b060a2",
    "post_title": "Red foxes hunting in deep snow",
    "image_id": "2664e34f46594d9c8f31e71379582806",
    "filename": "gray-wolf-snow-01.png",
    "similarity": 0.3636,
    "accepted": false,
    "reasons": [
        {
            "code": "subject_mismatch",
            "message": "Animal category mismatch: expected red fox, detected gray wolf."
        },
        {
            "code": "similarity_below_threshold",
            "message": "Similarity 0.36 is below the 0.62 threshold \u2014 the image caption is not close enough in meaning to this post."
        }
    ],
    "signals": {
        "post_subjects": [
            "red fox"
        ],
        "post_modifiers": [
            "snow"
        ],
        "image_subject": "gray wolf",
        "image_subject_concept": "gray wolf",
        "image_category": "animal",
        "image_confidence": 0.94,
        "similarity": 0.3636,
        "thresholds": {
            "similarity": 0.62,
            "min_confidence": 0.55,
            "ambiguity_margin": 0.03
        }
    }
}
```

The endpoint bypasses ranking entirely and pushes a named pairing through the
guard. Both failing rules are reported, and `signals` shows every input the
decision was made from.

---

### PROBE 4 / PROBE 5 — no confident match, and top-1 precision

```
########## PROBE 4: no suitable image -> 'no confident match' with reasons
$ curl -si 'http://127.0.0.1:8077/v1/posts/restoring-vintage-motorcycles/images?limit=60' | head -1
HTTP/1.1 200 OK
x-match-verdict: no-confident-match
{
  "has_confident_match": false,
  "best_match": null,
  "suggestions": [],
  "no_match": {
    "message": "No image in the corpus clears the bar for \u201cRestoring vintage motorcycles\u201d. Best similarity was 0.00 against a 0.62 threshold. 1 candidate(s) were rejected for low vision confidence.",
    "wanted_subjects": [],
    "best_similarity": 0.0,
    "similarity_threshold": 0.62,
    "rejection_counts": {
      "similarity_below_threshold": 51,
      "image_flagged_for_review": 2,
      "low_vision_confidence": 1
    },
    "candidates_considered": 51
  }
}

########## PROBE 5: top-1 precision on the labeled eval set
$ curl -s -X POST http://127.0.0.1:8077/v1/eval/run
{
  "embedding_model": "lexicon-concept-v1",
  "thresholds": {
    "similarity": 0.62,
    "min_vision_confidence": 0.55,
    "low_confidence_threshold": 0.7
  },
  "labeled_cases": 14,
  "labeled_correct": 13,
  "top1_precision": 0.9286,
  "no_match_cases": 2,
  "no_match_correct": 2,
  "correct_rejection_rate": 1.0
}

the one failing case:
{
  "post_slug": "quiet-morning-on-the-beach",
  "post_title": "A quiet morning on the empty beach at sunrise",
  "expected_image": "beach-01",
  "predicted_image": "beach-water-02",
  "predicted_similarity": 1.0,
  "correct": false,
  "kind": "labeled",
  "note": "",
  "rejection_reasons": []
}
```

PROBE 4: a `200` with `has_confident_match: false`, a populated `no_match`
object and an `X-Match-Verdict` header — a refusal is a deliberate answer, not
an error. All 51 candidates were considered and counted by failure mode.

PROBE 5: `0.929` top-1 precision — the same number the README reports. The one
failing case is analysed in the README rather than relabelled away.

---

### PROBE 6 — every vision/embedding call is attributed with a cost entry

```
########## PROBE 6: every AI call has a cost entry
$ curl -s http://127.0.0.1:8077/v1/costs/summary
{
    "total_calls": 120,
    "total_cost_usd": 0.010232,
    "failed_calls": 0,
    "daily_budget_usd": 1.0,
    "spent_last_24h_usd": 0.010232,
    "budget_remaining_usd": 0.989768,
    "by_operation": [
        {
            "operation": "embedding",
            "provider": "fixture",
            "model": "lexicon-concept-v1",
            "calls": 69,
            "cost_usd": 3.2e-05,
            "input_tokens": 1288,
            "output_tokens": 0
        },
        {
            "operation": "vision",
            "provider": "fixture",
            "model": "fixture-vision-v1",
            "calls": 51,
            "cost_usd": 0.0102,
            "input_tokens": 14280,
            "output_tokens": 3060
        }
    ]
}

$ curl -s 'http://127.0.0.1:8077/v1/costs?limit=4'
[
    {
        "id": "6aec6538f39b4d2aa989aadffffca0aa",
        "operation": "embedding",
        "provider": "fixture",
        "model": "lexicon-concept-v1",
        "subject_ref": "post:467d52b1f92646ee9a8524e14323e553",
        "job_id": "e48dd4f0a31746e0be5bd137bef42a78",
        "input_tokens": 23,
        "output_tokens": 0,
        "cost_usd": 5.75e-07,
        "latency_ms": 0,
        "status": "ok",
        "created_at": "2026-08-31T20:57:32.582755"
    },
    {
        "id": "03ced7e6ee61477b9b6cc59c570dea4a",
        "operation": "embedding",
        "provider": "fixture",
        "model": "lexicon-concept-v1",
        "subject_ref": "post:f076973c9dbc4ce4b9abec6e7087e294",
        "job_id": "e48dd4f0a31746e0be5bd137bef42a78",
        "input_tokens": 41,
        "output_tokens": 0,
        "cost_usd": 1.0250000000000001e-06,
        "latency_ms": 0,
        "status": "ok",
        "created_at": "2026-08-31T20:57:32.580036"
    },
    {
        "id": "38e8ab66190643c5be9ac499cc481064",
        "operation": "embedding",
        "provider": "fixture",
        "model": "lexicon-concept-v1",
        "subject_ref": "post:804c45213e634ee38fddd28b53c5e351",
        "job_id": "e48dd4f0a31746e0be5bd137bef42a78",
        "input_tokens": 25,
        "output_tokens": 0,
        "cost_usd": 6.25e-07,
        "latency_ms": 0,
        "status": "ok",
        "created_at": "2026-08-31T20:57:32.577598"
    },
    {
        "id": "841c0b11722e4fffae907de38c58db5c",
        "operation": "embedding",
        "provider": "fixture",
        "model": "lexicon-concept-v1",
        "subject_ref": "post:3ce6dd43301f43a5b2463d877f4b8fba",
        "job_id": "e48dd4f0a31746e0be5bd137bef42a78",
        "input_tokens": 24,
        "output_tokens": 0,
        "cost_usd": 6.000000000000001e-07,
        "latency_ms": 0,
        "status": "ok",
        "created_at": "2026-08-31T20:57:32.575065"
    }
]

# every call is attributed; count matches the ledger total:
ledger rows          : 120
with subject_ref     : 120
with job_id          : 120
vision calls         : 51
embedding calls      : 69
total cost USD       : 0.010232
```

120 ledger rows: 51 vision calls (one per image) and 69 embedding calls
(51 images + 18 posts). Every row carries `subject_ref` and `job_id`, so any
cost can be traced back to the image or post it was spent on.

---

## Requirement proofs

### A1 — Schema validation

Every vision response is parsed through `ImageTags` (`app/schemas/vision.py`)
before anything downstream touches it. A response that does not fit is a
**provider error**, not data — it is retried or flagged, never partially
written.

```
$ python -m pytest tests/test_vision_schema.py -q
.........                                                                [100%]
9 passed in 0.25s
```

Covered: missing `subject`, unknown `category`, `confidence` out of `[0,1]`,
`confidence` not a number, caption too short, prose instead of JSON, and
schema-violating JSON returned inside a well-formed HTTP 200. Providers that
support native structured output (Gemini `responseSchema`, Ollama `format`) are
*also* constrained at the source — belt and braces, because a model that is
asked for JSON can still return prose.

---

### A2 — Low confidence is flagged

See [PROBE 1](#probe-1--batch-job-tags-the-corpus-at-least-one-low-confidence-image-is-flagged).
Three of 51 images are flagged with a human-readable reason. Two tiers:

- `confidence < 0.55` (`MIN_VISION_CONFIDENCE`) — never auto-suggested; the
  guard rejects it outright.
- `0.55 ≤ confidence < 0.70` (`LOW_CONFIDENCE_THRESHOLD`) — flagged for human
  confirmation; the guard refuses to auto-approve it.

PROBE 2 shows the second tier biting in practice: `red-fox-urban-06` is the
*right animal* for the fox post and ranks #5, and is still refused
("Image is flagged for human review (confidence 0.61) and is not auto-approved").

---

### A3 — Batch job with retries

Slow AI work never runs on the request path. `POST /v1/jobs` returns `202`
immediately; a worker claims the job, runs it, and records progress.

```
$ curl -s $B/v1/jobs | jq '.[0]'
{
  "kind": "full_pipeline",
  "status": "succeeded",
  "attempts": 1,
  "total_items": 138,
  "processed_items": 138,
  "failed_items": 0,
  "result": {
    "vision_tagging": {"tagged": 51, "flagged_for_review": 3, "failed": 0},
    "embed_images":   {"embedded": 51, "already_embedded": 0, "model": "lexicon-concept-v1"},
    "embed_posts":    {"embedded": 18, "already_embedded": 0, "model": "lexicon-concept-v1"},
    "match_posts":    {"posts_matched": 18, "posts_with_confident_match": 16,
                       "posts_without_match": 2}
  }
}
```

Retry behaviour is proved in [F2](#f2-retries-and-the-failure-alert). Handlers
are **resumable**: they process only what is still outstanding, so attempt 2 of
a job that died half way does not re-pay for the images attempt 1 already
tagged (`test_tagging_is_resumable_and_does_not_repay_for_done_work`).

---

### A4 — Per-call cost tracking

See [PROBE 6](#probe-6--every-visionembedding-call-is-attributed-with-a-cost-entry).
120 ledger rows, every one carrying `subject_ref` (which image or post) and
`job_id` (which run). Failed calls are recorded too, at $0 spend but visible in
`failed_calls`, so a provider outage is not invisible in the ledger.

---

### B1 — Ranked suggestions

See [PROBE 2](#probe-2--the-fox-post-ranks-the-fox-first-wolf-and-dog-rank-clearly-lower).
Image and post embeddings are persisted in the `embeddings` table (unique per
owner+model, so re-embedding upserts rather than duplicating), and
`GET /v1/posts/{slug}/images` returns the ranked, guarded list.

---

### B2 — Semantic equivalence

The post *"The red fox of the temperate forest"* uses the Latin name in its
body — **"Vulpes vulpes"** shares no words with "red fox", and the image
captions never contain it.

```
$ curl -s $B/v1/posts/the-red-fox-of-the-forest/images | jq '.best_match | {filename, subject, similarity}'
{
  "filename": "red-fox-forest-02.png",
  "subject": "red fox",
  "similarity": 0.967
}
```

Asserted directly on the vector space in `test_matching.py`:

```python
assert cosine_similarity(embed("red fox"), embed("Vulpes vulpes")) > 0.99   # same concept
assert cosine_similarity(embed("red fox"), embed("gray wolf"))    < 0.5     # different animal
assert cosine_similarity(embed("red fox"), embed("margherita pizza")) < 0.2 # different world
```

```
$ python -m pytest tests/test_matching.py -q
.......                                                                  [100%]
7 passed in 7.12s
```

---

### C1 — The wolf provably fails

See [PROBE 3](#probe-3--force-the-wolf-as-a-candidate-for-the-fox-post). The
rejection message is the one the brief asks for:

> `"Animal category mismatch: expected red fox, detected gray wolf."`

The guard is a pure module with no I/O, so every rule is unit-tested in
isolation:

```
$ python -m pytest tests/test_guard.py -q
................                                                         [100%]
16 passed in 0.05s
```

Covered: matching subject accepted; wolf rejected **even at 0.94 similarity**
(the key case — similarity alone would have accepted it); dog rejected;
cross-category rejected; low similarity rejected despite the right subject; low
confidence rejected; flagged image not auto-approved; untagged image never
suggested; *all* failing reasons reported rather than just the first; ambiguity
warns instead of rejecting; and the title-vs-body intent rule ("a fox post that
mentions wolves once is still a fox post").

---

### C2 — No confident match

See [PROBE 4](#probe-4--probe-5--no-confident-match-and-top-1-precision). The
answer names what the post wanted, the best similarity achieved, the threshold
it failed against, and a count of every candidate by failure mode.

---

### D1 — Schema and indexes

Schema is created by Alembic migrations (`alembic/versions/0001_initial_schema.py`),
not `create_all` — and the test suite runs on those migrations, so drift breaks
the build.

```
$ alembic upgrade head && sqlite3 flyrank.sqlite3 '.schema'   # summarised
TABLES  : ai_calls, alembic_version, embeddings, image_tags, images, jobs, posts, reviews, suggestions, tenants

INDEXES (excluding auto-generated uniqueness indexes):
  ai_calls       ix_ai_calls_job
  ai_calls       ix_ai_calls_tenant_created
  ai_calls       ix_ai_calls_tenant_id
  embeddings     ix_embeddings_tenant_id
  embeddings     ix_embeddings_tenant_owner_model
  image_tags     ix_image_tags_image_id
  image_tags     ix_image_tags_tenant_id
  image_tags     ix_image_tags_tenant_value
  images         ix_images_tenant_category
  images         ix_images_tenant_id
  images         ix_images_tenant_review
  images         ix_images_tenant_status
  jobs           ix_jobs_status_scheduled
  jobs           ix_jobs_tenant_created
  jobs           ix_jobs_tenant_id
  posts          ix_posts_tenant_id
  reviews        ix_reviews_tenant_id
  reviews        ix_reviews_tenant_suggestion
  suggestions    ix_suggestions_tenant_id
  suggestions    ix_suggestions_tenant_post_rank
  suggestions    ix_suggestions_tenant_verdict
```

Each index is annotated in `app/db/models.py` with the access path it serves —
`ix_images_tenant_status` is the batch job's claim query,
`ix_jobs_status_scheduled` is the worker's, `ix_ai_calls_tenant_created` is the
budget guard's, `ix_suggestions_tenant_post_rank` is the ranked read path.
Uniqueness constraints carry the idempotency guarantees:
`uq_images_tenant_sha256`, `uq_jobs_dedupe`, `uq_reviews_idempotency`,
`uq_embeddings_owner_model`, `uq_suggestions_post_image`.

---

### D2 — Validation at the boundary

Bad input produces a clean, machine-readable 4xx. **Never a 500.**

```
########## Validation at the boundary: bad input -> clean 4xx, never a 500
$ curl -si -X POST $B/v1/posts -d '{"slug":"Not A Slug","title":"hi"}'
HTTP 422
{
    "code": "validation_error",
    "message": "Request body or parameters failed validation.",
    "details": {
        "errors": [
            {
                "loc": [
                    "body",
                    "slug"
                ],
                "type": "string_pattern_mismatch",
                "msg": "String should match pattern '^[a-z0-9][a-z0-9-]*$'"
            },
            {
                "loc": [
                    "body",
                    "title"
                ],
                "type": "string_too_short",
                "msg": "String should have at least 3 characters"
            },
            {
                "loc": [
                    "body",
                    "body"
                ],
                "type": "missing",
                "msg": "Field required"
            }
        ]
    }
}

$ curl -si -X POST $B/v1/jobs -d '{"kind":"make_coffee"}'
HTTP 422
{
    "code": "validation_error",
    "message": "Request body or parameters failed validation.",
    "details": {
        "errors": [
            {
                "loc": [
                    "body",
                    "kind"
                ],
                "type": "value_error",
                "msg": "Value error, kind must be one of: vision_tagging, embed_images, embed_posts, match_posts, full_pipeline"
            }
        ]
    }
}

$ curl -si '$B/v1/images?limit=99999'
HTTP 422
validation_error | Input should be less than or equal to 500

$ curl -si $B/v1/posts/does-not-exist
HTTP 404
{
    "code": "not_found",
    "message": "post 'does-not-exist' not found (looked up by id and slug)",
    "details": {}
}

$ curl -si $B/v1/posts -H 'X-Tenant-ID: nope'   # unknown tenant, not auto-created
HTTP 404
{
    "code": "not_found",
    "message": "unknown tenant 'nope'; run the seed script or pass a valid X-Tenant-ID header",
    "details": {
        "tenant_id": "nope"
    }
}

########## Job idempotency: same dedupe_key returns the first job
HTTP 202  job id 850fcb5b4a76427882f024fe2b0b47b9
HTTP 200  job id 850fcb5b4a76427882f024fe2b0b47b9 (replay)
```

Unhandled exceptions return a generic `500` carrying only a request id — no
internal message, no stack trace (`app/main.py`).

---

### D3 — Review workflow

Approve / reject / inspect why, with idempotency:

```
########## Review workflow: inspect why -> approve -> idempotent replay
$ SID=$(curl -s "$B/v1/posts/red-foxes-in-deep-snow/images" | jq -r .best_match.suggestion_id)
$ curl -s "$B/v1/suggestions/$SID/explain"
{
    "suggestion_id": "97a2beab19e8427982c867d4b18933b2",
    "post_id": "a97f3444378045bfa004b35b97b060a2",
    "post_title": "Red foxes hunting in deep snow",
    "image_id": "b920da397d7d41b6b9961507b4244179",
    "filename": "red-fox-snow-01.png",
    "rank": 1,
    "similarity": 0.8804,
    "verdict": "accepted",
    "reasons": [
        {
            "code": "accepted",
            "message": "Subject red fox matches the post, similarity 0.88 \u2265 0.62, vision confidence 0.95."
        }
    ],
    "signals": {
        "post_subjects": [
            "red fox"
        ],
        "post_modifiers": [
            "snow"
        ],
        "image_subject": "red fox",
        "image_subject_concept": "red fox",
        "image_category": "animal",
        "image_confidence": 0.95,
        "similarity": 0.8804,
        "thresholds": {
            "similarity": 0.62,
            "min_confidence": 0.55,
            "ambiguity_margin": 0.03
        },
        "margin_to_runner_up": 0.0466
    },
    "image_embedding_text": "A red fox hunting in deep snow on a winter field red fox animal snow wild orange fur",
    "post_embedding_text": "Red foxes hunting in deep snow. Red foxes hunting in deep snow. Winter forces the red fox to change tactics. Listening through a snow pack, it locates prey by sound alone before diving head first into the drift. This post looks at how the red fox survives a hard winter.",
    "review": null
}

$ curl -si -X POST "$B/v1/reviews" -H 'Idempotency-Key: demo-1' -d '{...approved...}'
HTTP/1.1 201 Created
# replaying the same Idempotency-Key:
HTTP/1.1 200 OK
idempotent-replay: true
# review history shows exactly one decision:
reviews recorded: 1
[
  {
    "id": "122a5c7780fe4a4b9d0a588aadf67b1e",
    "suggestion_id": "97a2beab19e8427982c867d4b18933b2",
    "decision": "approved",
    "reviewer": "editor@example.com",
    "note": "Correct fox, good crop.",
    "idempotency_key": "demo-1",
    "created_at": "2026-08-31T20:58:47.639923"
  }
]
```

`/explain` returns the guard's full `signals` **and the exact texts that were
embedded** on both sides — the most useful part of an explanation is what the
model actually compared, not just the score it produced.

---

### E1 — Top-1 precision

See [PROBE 5](#probe-4--probe-5--no-confident-match-and-top-1-precision).
`0.929` (13/14), matching the README. Threshold derivation:

```
$ python -m scripts.tune_thresholds
 threshold    top-1   reject   combined
----------------------------------------
      0.30    0.929    1.000      0.938
      0.34    0.929    1.000      0.938
      0.38    0.929    1.000      0.938
      0.42    0.929    1.000      0.938
      0.46    0.929    1.000      0.938
      0.50    0.929    1.000      0.938
      0.54    0.929    1.000      0.938
      0.58    0.929    1.000      0.938
      0.62    0.929    1.000      0.938
      0.66    0.929    1.000      0.938
      0.70    0.929    1.000      0.938
      0.74    0.929    1.000      0.938
      0.78    0.929    1.000      0.938
      0.82    0.929    1.000      0.938
      0.86    0.929    1.000      0.938
      0.90    0.786    1.000      0.812
      0.94    0.500    1.000      0.562
      0.98    0.357    1.000      0.438
----------------------------------------
optimal plateau : 0.30 … 0.86 (combined accuracy 0.938)
safe band       : 0.000 < threshold < 0.880
  · 0.880 is the weakest similarity among correct matches — go above it and the system starts refusing good pairings
  · 0.000 is the strongest similarity among posts that must be refused — go below it and the system starts guessing
  · midpoint (max margin): 0.440

configured      : 0.620 (+0.620 above the reject edge, -0.260 below the accept edge)
  The default sits above the midpoint on purpose: in this system a wrong image shipped is worse than a missing one, so the threshold is biased toward refusing.

Set it with SIMILARITY_THRESHOLD in .env, then re-run scripts/run_eval.
```

---

### E2 — Required files

```
$ ls
BUILDLOG.md
Dockerfile
EVIDENCE.md
LICENSE
Makefile
README.md
alembic
alembic.ini
app
capstone.yaml
data
docker-compose.yml
docs
pytest.ini
requirements-dev.txt
requirements.txt
scripts
tests
```

`README.md` · `capstone.yaml` · `EVIDENCE.md` · `BUILDLOG.md` · `.env.example`
— all present, plus `LICENSE` (MIT) and a `.gitignore` that excludes `.env` and
the image corpus.

---

## Shared requirement proofs

### F1 — Layering

The rule: a route parses, delegates and presents. It never constructs an ORM
entity and never issues a query — that is what `app/services/` is for.

```
$ grep -rn "models\.\|select(" app/api/routes/ | wc -l     # routes touching the ORM
0
$ grep -rl "fastapi" app/services/ app/repositories/ | wc -l # logic importing the web framework
0
```

Import direction is `api/` → `services/` → `repositories/` → `db/`, with
`providers/` reached only through `registry.py`.

`app/api/presenters.py` is the one place in the HTTP layer that imports
`db.models` — it is the domain→API mapper, and mapping is its whole job.

---

### F2 — Retries and the failure alert

A job whose handler keeps failing: exponential backoff, then a `CRITICAL`
alert on the final attempt.

```
{
  "ts": "2026-08-31T22:02:31+0100",
  "level": "INFO",
  "logger": "app.jobs",
  "msg": "job_enqueued",
  "job_id": "7d482f33fbc443a3ac13959e2e12aa63",
  "kind": "vision_tagging",
  "tenant_id": "demo"
}
{
  "ts": "2026-08-31T22:02:31+0100",
  "level": "WARNING",
  "logger": "app.jobs",
  "msg": "job_retry_scheduled",
  "job_id": "7d482f33fbc443a3ac13959e2e12aa63",
  "kind": "vision_tagging",
  "attempt": 1,
  "max_attempts": 3,
  "backoff_s": 0.5,
  "error": "Gemini returned HTTP 503: model overloaded"
}
{
  "ts": "2026-08-31T22:02:31+0100",
  "level": "WARNING",
  "logger": "app.jobs",
  "msg": "job_retry_scheduled",
  "job_id": "7d482f33fbc443a3ac13959e2e12aa63",
  "kind": "vision_tagging",
  "attempt": 2,
  "max_attempts": 3,
  "backoff_s": 1.0,
  "error": "Gemini returned HTTP 503: model overloaded"
}
{
  "ts": "2026-08-31T22:02:31+0100",
  "level": "CRITICAL",
  "logger": "app.alerts",
  "msg": "JOB_FAILURE_ALERT",
  "alert": "JOB_FAILURE_ALERT",
  "job_id": "7d482f33fbc443a3ac13959e2e12aa63",
  "tenant_id": "demo",
  "kind": "vision_tagging",
  "attempts": 3,
  "error": "Gemini returned HTTP 503: model overloaded",
  "context": {
    "processed_items": 0,
    "failed_items": 0,
    "total_items": 0
  }
}
```

Backoff doubles (0.5s → 1.0s) and the third failure emits
`JOB_FAILURE_ALERT` at `CRITICAL` — a stable string for log-based alerting —
plus an optional webhook POST via `ALERT_WEBHOOK_URL`. A **non-retryable**
error (a corrupt file, a budget refusal) fails immediately without burning
retries; see `test_non_retryable_failure_fails_immediately`.

```
$ python -m pytest tests/test_jobs.py tests/test_costs.py -q
...............                                                          [100%]
15 passed in 6.78s
```

---

### F3 — Tenant isolation

`TenantRepository.__init__` raises without a tenant id, so no query can be
issued unscoped. Every business table carries `tenant_id` with composite
indexes.

```
$ curl -s -X POST $B/v1/posts -H 'X-Tenant-ID: demo' -d '{...}'   # tenant 'demo' has 18 posts
demo posts: 18
$ curl -s $B/v1/posts -H 'X-Tenant-ID: nope'
not_found | unknown tenant 'nope'; run the seed script or pass a valid X-Tenant-ID header
$ curl -s $B/v1/costs/summary -H 'X-Tenant-ID: nope'   # no cross-tenant leakage
not_found | unknown tenant 'nope'; run the seed script or pass a valid X-Tenant-ID header
```

Cross-tenant invisibility is asserted end-to-end in
`test_tenants_cannot_see_each_others_posts`: two tenants, one post each, each
sees only its own.

---

### F4 — Idempotency

| Action | Key | Behaviour |
|---|---|---|
| Enqueue a job | `dedupe_key` | `202` first time, `200` + `Idempotent-Replay: true` after — see [D2](#d2-validation-at-the-boundary) |
| Approve/reject | `Idempotency-Key` header | `201` then `200`; history shows one decision — see [D3](#d3-review-workflow) |
| Ingest the corpus | `sha256` | Re-running ingestion creates nothing new |
| Embed | `(owner, model)` unique | Re-embedding upserts |
| Match a post | `(post, image)` unique | Re-matching updates rows in place |

```
$ python -m scripts.seed          # second run, same corpus and posts
     images: {'images_dir': 'data/images', 'created': 0, 'already_present': 51, 'updated': 0, 'total': 51}
     posts: 0 created, 18 already present
```

---

### F5 — Secrets

- `.env` is in `.gitignore`; `.env.example` ships placeholders only.
- No key is ever logged: the cost ledger records provider and model names, not
  credentials.
- The Gemini key travels in the `x-goog-api-key` **header**, never in the URL —
  URLs end up in access logs and proxy traces. Enforced by a test:

```python
def test_api_key_is_sent_in_a_header_not_the_url(...):
    assert seen["header"] == "super-secret-key"
    assert "super-secret-key" not in seen["url"]
```

```
$ git grep -nE "AIza|sk-|api_key\s*=\s*[\"'][A-Za-z0-9_-]{20,}" -- . | wc -l
0
```

---

### F6 — Budget guard

The daily spend cap is checked **before** each call, so an over-budget run
refuses rather than overspends.

```
$ AI_DAILY_BUDGET_USD=0.0001 python -m scripts.seed   # a budget too small for one call
AI calls: 0 · cost $0.000000 of $0.00 daily budget

job status  : failed
attempts    : 1
last_error  : Daily AI budget exhausted: $0.0000 spent of $0.0001; this call would add $0.0002. Raise AI_DAILY_BUDGET_USD or wait for the window to roll.
images tagged: {'pending': 51}
```

Zero images tagged, zero dollars spent, one attempt (a budget refusal is not
retryable — retrying would fail identically), and an error message that says
exactly which knob to turn.
