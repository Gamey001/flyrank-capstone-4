# Design document

The Phase 1 one-pager: problem, data model, API surface, layer sketch, and one
explicit non-goal. Written before the build; kept accurate as it landed.

---

## Problem

Given ~50 images and a set of blog posts, suggest the right image for each post
**by meaning** — a red-fox post gets the red-fox photo, not the wolf, not a
generic dog.

The hard requirement is not retrieval. It is **refusal**: when the best
candidate is still wrong, the system must say so and say why, rather than
return its least-bad guess. A wolf caption and a fox post are genuinely close
in vector space, so cosine similarity alone will happily suggest the wolf. That
single fact drives the whole design.

**Success looks like:** fox post → fox image; wolf forced onto a fox post →
rejected with a category-mismatch reason; post about something absent from the
corpus → "no confident match" plus the reasons every candidate failed; and a
top-1 precision number measured on a hand-labeled set, not asserted.

---

## Data model

Four groups. Every business table carries `tenant_id`.

| Table | Holds | Key constraints |
|---|---|---|
| `tenants` | isolation boundary | — |
| `images` | file + **validated** vision metadata (`subject`, `category`, `caption`, `confidence`, `attributes`), `status`, `needs_review` | `UNIQUE(tenant, sha256)` — ingestion is idempotent on content |
| `image_tags` | normalised `(kind, value)` tags | `UNIQUE(image, kind, value)` — lets the guard do set logic without parsing JSON |
| `posts` | slug, title, body | `UNIQUE(tenant, slug)` |
| `embeddings` | one vector per `(owner, model)`, plus the text that was embedded | `UNIQUE(tenant, owner_type, owner_id, model)` — re-embedding upserts |
| `suggestions` | ranked `(post, image)` + guard `verdict`, `reasons`, `explanation` | `UNIQUE(tenant, post, image)` — re-matching updates in place |
| `reviews` | approve/reject decisions | `UNIQUE(tenant, idempotency_key)` — a retried approve lands once |
| `jobs` | queued work, attempts, progress, result | `UNIQUE(tenant, dedupe_key)` — a resubmitted job returns the original |
| `ai_calls` | the cost ledger: operation, model, `subject_ref`, `job_id`, tokens, cost, status | indexed on `(tenant, created_at)` for the budget guard |

Vectors are a JSON array, not pgvector: at ~50 images a full scan is
microseconds, and it keeps SQLite (tests, keyless demo) and Postgres identical.

**Indexes are chosen from the queries actually issued** — `(tenant, status)` for
the batch job's claim, `(status, scheduled_at)` for the worker's, `(tenant,
post, rank)` for the ranked read.

---

## API surface

```
POST /v1/images/ingest                       register corpus files (idempotent on sha256)
GET  /v1/images | /images/stats | /images/{id}   browse, filter by needs_review

POST /v1/posts        GET /v1/posts | /posts/{id-or-slug}
GET  /v1/posts/{ref}/images                  ★ ranked + guarded, or explained "no match"
POST /v1/posts/{ref}/images/{image}/check    ★ force one pairing through the guard

POST /v1/jobs   GET /v1/jobs | /jobs/{id}    background work, idempotent on dedupe_key
GET  /v1/suggestions | /suggestions/{id}/explain     the review queue, and *why*
POST /v1/reviews      GET /v1/reviews        approve/reject, idempotent on Idempotency-Key
GET  /v1/costs | /costs/summary              per-call ledger + remaining budget
POST /v1/eval/run                            top-1 precision on the labeled set
```

A refusal is a **200**, not an error: `has_confident_match: false` with a
populated `no_match` object. Refusing is a correct answer, and modelling it as
a 4xx would be a lie about what happened.

---

## Layer sketch

```
app/api/         HTTP: parse → delegate → present. Never touches the ORM.
   ↓
app/services/    the logic. Never imports FastAPI.
   guard.py      ★ pure: no DB, no I/O, no settings. Every rule unit-testable.
   matching.py   rank first, then guard every candidate
   jobs.py       queue policy: retries, backoff, alerting
   ↓
app/repositories/  tenant-scoped access. Cannot be constructed without a tenant.
   ↓
app/db/          SQLAlchemy models + Alembic migrations
app/providers/   gemini · ollama · offline fixtures, behind one Protocol
```

**Ranking and guarding stay separate steps, in that order.** The tempting
alternative — folding the tag check into the similarity score as a weight —
produces a low number and no explanation. Kept separate, it produces
*"expected red fox, detected gray wolf"*, which is the actual product.

The guard combines three independent signals, all of which must pass: vision
**confidence**, semantic **similarity**, and tag **agreement**. Only the third
knows that a fox and a wolf are different animals.

---

## Non-goal

**No image search engine, and no frontend.** There is no free-text query
endpoint, no faceted search, no pagination over a large corpus, and no UI. The
review interface is API endpoints. The system answers exactly one question —
*"is this image right for this post, and if not, why not?"* — and everything
else is deliberately out of scope.

Secondary exclusions, for the same reason: no authentication (`X-Tenant-ID` is
a scoping header, not a security boundary), no model comparison, and no vector
index beyond a full scan.
