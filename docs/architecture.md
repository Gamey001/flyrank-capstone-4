# Architecture notes

Supplementary detail behind the diagram in the README. Read that first.

## Request paths

### `GET /v1/posts/{slug}/images` — the read path

```
route (posts.py)
  └─ content.resolve_post()                     id or slug → Post
  └─ MatchingService.match_and_persist()
       ├─ EmbeddingService.ensure_post_embedding()   cached; embeds on first read
       ├─ EmbeddingRepository.vectors_by_owner()     all image vectors for the tenant
       ├─ rank_by_similarity()                       cosine, stable tie-break on filename
       ├─ guard.evaluate() per candidate             confidence · tags · similarity
       └─ SuggestionRepository.upsert()              reviewable rows, no duplicates
  └─ presenters.match_response()
```

Two properties worth noting:

- **The guard runs on every candidate**, not just the winner. That is what makes
  an explained refusal possible: `no_match_explanation()` aggregates why each
  of the 51 candidates failed.
- **Post embedding is lazy but image embedding is not.** Embedding one post is a
  single fast call, so doing it inline on first read is fine. Embedding 51
  images is bulk work and belongs in a job.

### `POST /v1/jobs` — the write path

The request only inserts a row and returns `202`. Nothing slow happens on the
request path.

```
route (jobs.py) → enqueue_job() → INSERT jobs(status='queued')
                                   ↑ UNIQUE(tenant_id, dedupe_key) — replay returns the original

worker loop  →  claim_next_job()          UPDATE ... status='running', attempts += 1
                                          FOR UPDATE SKIP LOCKED on Postgres
             →  HANDLERS[kind](...)       resumable: processes only what is outstanding
             →  run_job() applies the policy:
                   success            → status='succeeded', result=<summary>
                   retryable + left   → status='queued', scheduled_at = now + backoff
                   retryable + spent  → status='failed'  + CRITICAL JOB_FAILURE_ALERT
                   non-retryable      → status='failed'  + alert, no retries burned
```

`full_pipeline` chains `vision_tagging → embed_images → embed_posts →
match_posts` inside one job so a demo is one call. Because each handler is
resumable, a mid-chain failure that retries does not re-pay for completed work.

## Why the queue is a table

At this scale a broker is a service to run, a client library to learn, and a
second place for state to live. A table gives:

- job history queryable alongside the data it produced (`ai_calls.job_id`);
- idempotency as a database constraint rather than application logic;
- correct multi-worker claim via `FOR UPDATE SKIP LOCKED`;
- one fewer container.

What it does not give: priorities, fan-out, cross-process scheduling
guarantees, or backpressure. Those are the reasons to switch, and none of them
apply to 51 images.

## Failure semantics

| Failure | Where it surfaces | Behaviour |
|---|---|---|
| Model returns prose instead of JSON | `SchemaValidationError` | retryable; image `status='failed'`, `last_error` set; job retries |
| Model returns JSON that breaks the schema | `SchemaValidationError` | same — a well-formed 200 carrying bad tags is still a failure |
| Provider HTTP 5xx / 429 | `ProviderError(retryable=True)` | job retries with backoff |
| Provider HTTP 4xx (bad key, bad request) | `ProviderError(retryable=False)` | job fails immediately; retrying would fail identically |
| Image file missing or corrupt | `ProviderError(retryable=False)` | that image fails; the batch continues |
| Daily budget exhausted | `BudgetExceededError` | whole batch stops, no retry — continuing would fail every remaining item and fill the ledger with noise |
| Worker crashes mid-job | row left `running` | `reclaim_stale_jobs()` requeues it on next start |

## Tenancy

`tenant_id` is on every business table, with composite indexes leading on it.
`TenantRepository.__init__` raises without one, so an unscoped query cannot be
written by accident — isolation is a property of the data-access layer rather
than something each endpoint must remember.

`claim_next_job` is the one deliberate exception: the worker serves all
tenants. Isolation survives because the job row carries its own `tenant_id` and
the handler builds tenant-scoped repositories from it.

**This is scoping, not security.** `X-Tenant-ID` is unauthenticated. A real
deployment would derive the tenant from a verified token.

## The embedding space

`image_embedding_text()` deliberately excludes the filename. Matching on
`red-fox-snow-01.jpg` would be keyword search wearing a vector costume — the
whole point is that meaning, not naming, drives the match.

`post_embedding_text()` repeats the title once before the body. The title
carries the post's intent far more reliably than the body, and a long body
would otherwise drown it. This is a cheap, honest weighting; a real system
would use a title-aware model or separate title/body vectors.

## Scaling notes

| Component | Fine until | Then |
|---|---|---|
| Vector scan (`vectors_by_owner` + `rank_by_similarity`) | ~10k images | pgvector + an ANN index; the change is confined to those two functions |
| Table queue | a few workers, thousands of jobs/day | a broker |
| In-process worker (`WORKER_ENABLED=true`) | local dev, demos | the separate `worker` service (what docker-compose already runs) |
| Lazy post embedding on read | posts created one at a time | precompute in `embed_posts` |
