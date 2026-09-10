# AI Image Understanding & Content Matching Engine

A backend that looks at an image library, understands what is actually *in*
each image, and matches each image to the right blog post — by meaning, not by
filename or keyword.

The point of the system is not finding a match. It is **refusing a wrong one**:

```
Post:      "Red foxes hunting in deep snow"
Candidate: "A gray wolf crossing a snowfield in winter"
Similarity: 0.36
Result:    REJECTED
Reason:    "Animal category mismatch: expected red fox, detected gray wolf."
```

Good suggestions when confident, an explained refusal when not.

**Headline quality number: top-1 precision `0.929` (13/14 labeled posts), with
`1.00` correct-rejection rate on the posts whose right answer is "no match".**
Reproduce it with `python -m scripts.run_eval` — see
[How the number is measured](#evaluation-how-the-number-is-measured).

---

## Table of contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The mismatch guard](#the-mismatch-guard)
- [Running against a real AI model](#running-against-a-real-ai-model)
- [The corpus](#the-corpus)
- [Evaluation](#evaluation-how-the-number-is-measured)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations](#limitations--what-i-would-do-next)

---

## Quick start

### Docker (the one-command path)

```bash
docker compose up --build -d
docker compose exec api sh -c "python -m scripts.make_placeholder_corpus && python -m scripts.seed"
curl -s localhost:8000/v1/posts/red-foxes-in-deep-snow/images | jq
```

That boots Postgres, applies the migrations, starts the API and a separate
worker process, builds the corpus, and runs vision tagging → embeddings →
matching. **No API key and no network access are required** — see
[providers](#running-against-a-real-ai-model).

### Local (Python 3.11+, SQLite, no services)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env

.venv/bin/python -m scripts.make_placeholder_corpus   # or: -m scripts.fetch_corpus
.venv/bin/python -m scripts.seed
.venv/bin/uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

### See it work

```bash
# the fox post gets a fox; the wolf and the dog rank lower and are refused
curl -s 'localhost:8000/v1/posts/red-foxes-in-deep-snow/images?limit=60' | jq

# force the wolf onto the fox post — the guard refuses it with a reason
WOLF=$(curl -s 'localhost:8000/v1/images?limit=200' \
  | jq -r '.[] | select(.filename|startswith("gray-wolf-snow-01")) | .id')
curl -s -X POST "localhost:8000/v1/posts/red-foxes-in-deep-snow/images/$WOLF/check" | jq

# a post about something the corpus does not contain
curl -s localhost:8000/v1/posts/restoring-vintage-motorcycles/images | jq .no_match

# the cost ledger
curl -s localhost:8000/v1/costs/summary | jq
```

---

## Architecture

Two embedding streams meet at the ranking step, and **everything passes the
guard before a human sees it**.

```
                     ┌──────────────────── background jobs (retries, progress, cost) ───────────────────┐
                     │                                                                                  │
  data/images/       │   ┌───────────────┐      ┌──────────────────────┐      ┌──────────────────┐      │
  ~51 photos  ──ingest──▶│ vision model  │─────▶│ ImageTags (Pydantic) │─────▶│ images           │      │
  (sha256 dedupe)    │   │ Gemini/Ollama │ JSON │ schema-validated     │      │ + image_tags     │      │
                     │   └───────────────┘      │ INVALID ⇒ retry/flag │      │ low conf ⇒ flag  │      │
                     │                          └──────────┬───────────┘      └────────┬─────────┘      │
                     │                                     │ embed(caption+tags)       │                │
                     │                                     ▼                           │                │
                     │                            ┌──────────────────┐                 │                │
  posts ─────────────┼──── embed(title+body) ────▶│ embeddings       │◀────────────────┘                │
                     │                            │ (shared space)   │                                  │
                     └────────────────────────────┴────────┬─────────┴──────────────────────────────────┘
                                                           │
                              GET /v1/posts/{slug}/images   │
                                                           ▼
                                            ┌──────────────────────────────┐
                                            │  1. cosine similarity rank   │   "which is closest?"
                                            └───────────────┬──────────────┘
                                                            ▼
                                            ┌──────────────────────────────┐
                                            │  2. MISMATCH GUARD           │   "is the closest one right?"
                                            │     · vision confidence      │
                                            │     · similarity threshold   │
                                            │     · tag/subject agreement  │
                                            └───────┬──────────────┬───────┘
                                                    │              │
                                       accepted ────┘              └──── refused
                                                    │                        │
                                        ranked suggestion            "no confident match"
                                        + why it was chosen          + why every candidate failed
                                                    │                        │
                                                    └────────┬───────────────┘
                                                             ▼
                                          Review API: approve / reject / explain
```

### Layers

| Layer | Directory | Rule |
|---|---|---|
| HTTP | `app/api/` | Parse, delegate, present. Never touches the ORM. |
| Logic | `app/services/` | The pipeline, the guard, jobs, costs, eval. Never imports FastAPI. |
| Data | `app/repositories/` | Tenant-scoped access. A repository cannot be built without a tenant. |
| Persistence | `app/db/` | SQLAlchemy models + Alembic migrations. |
| External AI | `app/providers/` | Gemini, Ollama, and deterministic offline fixtures behind one Protocol. |

The Phase 1 one-pager — problem, data model, API surface, layer sketch and the
explicit non-goal — is [docs/design.md](docs/design.md). Deeper notes on request
paths, failure semantics, why the queue is a table and where it stops scaling
are in [docs/architecture.md](docs/architecture.md).

### Why ranking and guarding are separate steps

Cosine similarity answers *"which image is closest?"*. It cannot answer *"is
the closest image actually correct?"* — and on this corpus those are genuinely
different questions. A wolf caption and a fox post are semantically close; the
ranker is right to score the wolf highly. Only a tag-level check knows that a
fox and a wolf are different animals. Running the guard **after** ranking, on
every candidate rather than just the winner, is what lets the API return an
*explained refusal* instead of an empty list.

---

## The mismatch guard

`app/services/guard.py` is a pure module: no database, no I/O, no settings
import. Everything it needs arrives as arguments, which makes every rule
directly unit-testable and the thresholds tunable from the eval set.

| # | Signal | Rejects when | Reason code |
|---|---|---|---|
| 0 | Tagged | the image has no validated metadata | `image_not_tagged` |
| 1 | Confidence | the vision model was below `MIN_VISION_CONFIDENCE` | `low_vision_confidence` |
| 1b | Review flag | the image is flagged for a human and not auto-approved | `image_flagged_for_review` |
| 2 | Tag agreement | the post asks for subject X, the image shows subject Y | `subject_mismatch` / `category_mismatch` |
| 3 | Similarity | cosine below `SIMILARITY_THRESHOLD` | `similarity_below_threshold` |
| 4 | Ambiguity | top two candidates within `AMBIGUITY_MARGIN` | *warning only, not a rejection* |

Every failing rule is reported, not just the first — a reviewer should see
every reason a pairing was refused. Rule 4 deliberately warns rather than
rejects: a close call is worth a human glance, not an automatic refusal.

### Where the thresholds come from

`python -m scripts.tune_thresholds` sweeps the similarity threshold against the
labeled eval set and reports the decision boundary:

```
optimal plateau : 0.30 … 0.86 (combined accuracy 0.938)
safe band       : 0.000 < threshold < 0.880
  · 0.880 is the weakest similarity among correct matches — go above it and the system starts refusing good pairings
  · 0.000 is the strongest similarity among posts that must be refused — go below it and the system starts guessing
  · midpoint (max margin): 0.440

configured      : 0.620 (+0.620 above the reject edge, -0.260 below the accept edge)
```

Two honest observations:

1. **The plateau is wide** because the tag-agreement rule already refuses
   wrong-subject candidates before similarity is consulted. The similarity
   threshold is the *second* line of defence, for posts about subjects the
   corpus does not contain at all.
2. **`0.62` sits above the midpoint on purpose.** In this system a wrong image
   shipped is worse than a missing one, so the threshold is biased toward
   refusing. That is a product decision, and it is written down rather than
   buried.

---

## Running against a real AI model

Every requirement maps to a tool that is free with **no credit card**. Three
provider paths, selected by environment variable, with an identical pipeline:

| `VISION_PROVIDER` / `EMBEDDING_PROVIDER` | What it is | Needs |
|---|---|---|
| `stub` *(default)* | Deterministic offline fixtures | nothing |
| `gemini` | Gemini Flash + `text-embedding-004`, free tier | a free key from [AI Studio](https://aistudio.google.com/apikey) |
| `ollama` | `llava` + `all-minilm`, fully local | `ollama serve` |

```bash
# Gemini Flash free tier
echo "GEMINI_API_KEY=..." >> .env
VISION_PROVIDER=gemini EMBEDDING_PROVIDER=gemini python -m scripts.seed

# fully local, offline
ollama pull llava && ollama pull all-minilm
VISION_PROVIDER=ollama EMBEDDING_PROVIDER=ollama python -m scripts.seed
```

### About the default `stub` provider — read this

The default providers are **fixtures, not models**, and the README would be
dishonest if it implied otherwise:

- `FixtureVisionProvider` replays the ground truth from
  `data/corpus/manifest.json` in the same validated shape a real model returns,
  including deliberately low confidence on three images.
- `LexiconEmbeddingProvider` projects text onto the concept lexicon in
  `app/services/concepts.py`. It is a real vector space with real cosine
  behaviour — "red fox" and "Vulpes vulpes" land on the same axis — but its
  vocabulary is that lexicon, not the language.

They exist so a reviewer can clone the repo and watch the whole pipeline —
batch tagging, ranking, the guard refusing the wolf, the eval number — with no
key, no network and no GPU, and so the test suite is deterministic. **The
headline precision number was measured with them.** With Gemini embeddings the
similarity distribution is different and the threshold must be re-tuned; the
tuner exists for exactly that.

---

## The corpus

**51 images across 4 categories and 12 subjects** — animals (red fox, gray
wolf, dog, brown bear, deer), landscape (forest, mountain, beach), food (pizza,
coffee), architecture (bridge, cathedral). Large enough to show real retrieval
behaviour, small enough to process cheaply and check by eye.

The image files are **not committed** — a few MB of photos does not belong in a
git repository. `data/corpus/manifest.json` defines the corpus, and two scripts
build it:

```bash
python -m scripts.fetch_corpus              # real CC-licensed photos (needs network)
python -m scripts.make_placeholder_corpus   # deterministic offline tiles (no network)
```

`fetch_corpus` pulls from [Openverse](https://openverse.org) — free, no key, no
card — accepting only permissive licences (CC0, Public Domain Mark, CC BY) and
writing the full attribution trail to `data/corpus/attribution.json`, so the
licence record lives in the repo even though the bytes do not.

`make_placeholder_corpus` writes one small deterministic PNG per manifest entry.
These are flat colour tiles, not photographs: they exercise the plumbing, and
they pair with `VISION_PROVIDER=stub`. A real vision provider pointed at them
would — correctly — describe coloured squares.

The two scripts compose safely: a download supersedes a placeholder of the same
slug, and the generator never shadows a real photo, so no image is ever
ingested twice under two extensions.

Two caveats on the download path, since both matter for interpreting results:

- **Openverse search is keyword search.** Most queries return the right subject,
  but not all — one run returned a barn for `"red fox meadow"`. Skim
  `data/corpus/attribution.json` and re-run individual entries before trusting
  a fetched corpus.
- **`fetch_corpus` is for use with a real vision provider.** The `stub` provider
  replays the manifest's ground truth regardless of what the file actually
  shows, so pairing downloaded photos with `VISION_PROVIDER=stub` would label
  them from the manifest rather than from the image. Fetch photos *and* set
  `VISION_PROVIDER=gemini` (or `ollama`) together.

---

## Evaluation: how the number is measured

`data/seed/eval_set.json` is a hand-labeled set of **16 posts**: 14 where a
human named the single correct image, and 2 whose correct answer is a refusal.

```bash
python -m scripts.run_eval --json eval/report.json
```

```
TOP-1 PRECISION      : 0.929  (13/14 labeled posts)
CORRECT REJECTION    : 1.000  (2/2 no-match posts)
```

The two populations are scored separately on purpose. Counting the no-match
posts as precision failures would reward a system that guesses; scoring them as
*correct rejections* measures the thing this capstone is actually about.

### The one failing case, and why it stays

```
FAIL  quiet-morning-on-the-beach   expected=beach-01  got=beach-water-02  sim=1.000
```

The post is *"A quiet morning on the empty beach at sunrise"* and its body says
"the sea is flat". The lexicon embedding reads "sea" as the `water` concept,
which pulls the ranking to the wave photo instead of the empty-sand photo. Both
are beaches; the guard correctly accepts either. This is a real weakness of a
bag-of-concepts embedding — it has no notion of which concept the sentence is
*about* — and it is left in rather than papered over by relabelling. A real
embedding model handles this case; the lexicon is the price of running with no
API key.

`scripts/run_eval.py` exits non-zero below 0.8, so it works as a CI quality
gate.

---

## API reference

Base URL `http://localhost:8000`, API prefix `/v1`. Full OpenAPI at `/docs`.
Every endpoint is tenant-scoped via the optional `X-Tenant-ID` header
(default `demo`); an unknown tenant is a `404`, never a silent new tenant.

### Corpus

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/images/ingest` | Register corpus files. Idempotent on content hash. |
| `GET` | `/v1/images` | Browse; filter by `status`, `category`, `needs_review`. |
| `GET` | `/v1/images/stats` | Counts by status plus the flagged-for-review count. |
| `GET` | `/v1/images/{id}` | One image with its validated metadata. |

### Posts and matching

| Method | Path | Purpose |
|---|---|---|
| `POST` `GET` | `/v1/posts` | Create / list posts. |
| `GET` | `/v1/posts/{id-or-slug}` | One post. |
| `GET` | `/v1/posts/{ref}/images` | **Ranked, guarded suggestions** — or an explained `no_match`. |
| `POST` | `/v1/posts/{ref}/images/{image_id}/check` | **Force one pairing through the guard**, bypassing ranking. |

A refusal is a deliberate, successful answer: `200` with
`has_confident_match: false`, a populated `no_match` object, and an
`X-Match-Verdict: no-confident-match` header.

### Review

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/suggestions` | The review queue. |
| `GET` | `/v1/suggestions/{id}/explain` | Why this image was chosen — or refused. Includes the exact texts that were embedded. |
| `POST` | `/v1/reviews` | Approve / reject. Idempotent via `Idempotency-Key`. |
| `GET` | `/v1/reviews` | Decision history. |

### Jobs and costs

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/jobs` | Queue `vision_tagging`, `embed_images`, `embed_posts`, `match_posts` or `full_pipeline`. Idempotent on `dedupe_key` (`202` new, `200` replay). |
| `GET` | `/v1/jobs` `/v1/jobs/{id}` | Status, attempts, per-item progress. |
| `GET` | `/v1/costs` | The per-call ledger — every call attributed to an image/post and a job. |
| `GET` | `/v1/costs/summary` | Spend by operation plus remaining daily budget. |
| `POST` | `/v1/eval/run` | Top-1 precision on the labeled set. |

### Production concerns, and where they live

| Concern | Implementation |
|---|---|
| Layered architecture | `api/` → `services/` → `repositories/` → `db/`, enforced by import direction |
| Validation at the boundary | Pydantic request models + typed handlers in `app/main.py`; bad input is `422`/`404`/`409`, never a `500`. Internal errors never leak a stack trace. |
| Background jobs | `app/services/jobs.py` + `app/worker.py`: table-backed queue, exponential backoff, resumable handlers, `CRITICAL JOB_FAILURE_ALERT` + optional webhook on final failure, stale-job reclaim after a crash |
| Real persistence | Alembic migrations (the test suite runs on them, not `create_all`); indexes chosen from the queries actually issued and annotated with the access path they serve |
| Idempotency | jobs on `dedupe_key`, reviews on `Idempotency-Key`, ingestion on `sha256`, embeddings and suggestions upsert rather than duplicate |
| Tenant isolation | `TenantRepository` cannot be constructed without a tenant; every business table carries `tenant_id` with composite indexes |
| Secrets | env only, `.env` gitignored, `.env.example` shipped; the Gemini key travels in a header, never in a URL (there is a test for that) |
| Cost tracking + budget guard | every call priced, attributed and logged — successes *and* failures; `AI_DAILY_BUDGET_USD` refuses a call rather than overspending |

---

## Project layout

```
app/
  api/routes/       HTTP: images, posts, jobs, review, costs, evaluation, health
  services/
    guard.py        ★ the mismatch guard — pure, no I/O
    concepts.py     ★ the concept lexicon (fox ≠ wolf; "Vulpes vulpes" = fox)
    matching.py     rank, then guard every candidate
    vision_pipeline.py  validate, flag low confidence, persist
    embedding_service.py, similarity.py, jobs.py, costs.py, evaluation.py,
    ingestion.py, alerts.py
  providers/        gemini.py · ollama.py · stub.py behind one Protocol
  repositories/     tenant-scoped data access
  db/               models + session
  schemas/          vision.py (the model contract) · api.py (the HTTP contract)
  main.py worker.py
alembic/versions/   migrations
data/
  corpus/manifest.json    the corpus definition + ground truth
  seed/posts.json         18 posts
  seed/eval_set.json      16 labeled eval cases
scripts/            fetch_corpus · make_placeholder_corpus · seed · run_eval · tune_thresholds
tests/              73 tests
docs/design.md        the Phase 1 one-pager (problem · model · API · non-goal)
docs/architecture.md  request paths, failure semantics, scaling notes

capstone.yaml  EVIDENCE.md  BUILDLOG.md  .env.example  docker-compose.yml
```

---

## Testing

```bash
python -m pytest          # 73 passed
```

Tests run against a throwaway SQLite database built **from the Alembic
migrations**, not `create_all`, so a migration that drifts from the models
fails the suite rather than production.

| File | Covers |
|---|---|
| `test_guard.py` (16) | every guard rule, including the wolf-on-a-fox-post rejection and its message |
| `test_vision_schema.py` (9) | schema validation, prose-instead-of-JSON, out-of-range confidence, key-not-in-URL |
| `test_matching.py` (7) | fox ranks first, wolf/dog rank lower, Latin-name matching, stable ordering |
| `test_jobs.py` (9) | retry with backoff, exhausted retries → alert, non-retryable fails fast, resumability, stale reclaim |
| `test_costs.py` (6) | attribution, budget guard, failed calls logged at $0 |
| `test_api.py` (26) | all six acceptance probes, 4xx validation, tenant isolation, review idempotency |

---

## Limitations — what I would do next

**Known and deliberate:**

- **The default providers are fixtures.** The headline precision number was
  measured with a bag-of-concepts embedding whose vocabulary is a hand-written
  lexicon. It is honest about equivalence ("Vulpes vulpes" ≈ "red fox") and
  weak about emphasis (the beach case above). Re-tune after switching to Gemini.
- **The eval set is 16 posts.** Enough to set a threshold and catch a
  regression; not enough for a confidence interval. 13/14 and 12/14 are one
  post apart.
- **The guard's tag check only knows the lexicon.** A subject outside
  `concepts.py` falls through to similarity alone. A real deployment would
  derive confusable-subject groups from the taxonomy rather than hand-listing
  them.
- **The vector "index" is a full scan** of a JSON column. At 51 images that is
  microseconds and avoids standing up pgvector. It stops being the right call
  somewhere around 10k images — the swap is confined to
  `EmbeddingRepository.vectors_by_owner` and `rank_by_similarity`.
- **The job queue is a table.** Correct for one worker and, with
  `FOR UPDATE SKIP LOCKED` on Postgres, for several. It is not a broker: no
  priorities, no fan-out, no cross-process scheduling guarantees.
- **The review API has no authentication.** Reviewer identity is whatever the
  caller claims. Deliberately out of scope; `X-Tenant-ID` is a scoping header,
  not a security boundary.
- **Placeholder images are not photographs.** Run `scripts/fetch_corpus.py`
  *and* switch to a real vision provider before drawing conclusions about model
  accuracy — and check the fetched corpus by eye, because Openverse search is
  keyword-based and occasionally returns the wrong subject.

**Next, in order:** re-tune thresholds against Gemini embeddings and report
both numbers side by side; grow the eval set past 40 posts; add alt-text
generation from the metadata already extracted; near-duplicate detection via
embedding distance.

---

## License

MIT — see [LICENSE](LICENSE).
