# BUILDLOG

An honest record of how this was built, where AI helped, where it was wrong,
and what I changed. Written as I went.

---

## Phase 1 — Design

**Decisions made before writing code:**

1. **The guard is a pure module.** The brief says "build the guard as its own
   small module first". I took that literally: `app/services/guard.py` has no
   database import, no settings import, no I/O. Everything it needs arrives as
   arguments. This paid off twice — every rule became directly unit-testable,
   and the threshold tuner could re-run the guard at 18 different thresholds
   without touching persistence.

2. **Ranking and guarding are separate steps, in that order.** The temptation
   is to fold the tag check into the similarity score as a weight. I didn't,
   because the two answer different questions and the *refusal reason* is the
   product. If the tag check is a weight, you get a low score and no
   explanation; kept separate, you get "expected red fox, detected gray wolf".

3. **Providers behind a Protocol, with an offline default.** The single most
   important usability decision. A reviewer clones the repo and sees the whole
   pipeline with no key, no network, no GPU — and the test suite is
   deterministic. Documented loudly in the README as fixtures, not models,
   because pretending otherwise would be dishonest.

4. **Two eval populations, scored separately.** Posts with a labeled correct
   image (top-1 precision) and posts whose correct answer is a refusal
   (correct-rejection rate). Folding the second into the first would reward a
   system that guesses — the exact failure mode the capstone is about.

---

## Where AI helped

- **Boilerplate volume.** SQLAlchemy models, Alembic setup, Pydantic API
  schemas, the FastAPI wiring, docker-compose. This is the work AI is
  genuinely good at, and it was faster than typing it.
- **The concept lexicon.** Generating the synonym lists (`vulpes vulpes`,
  `canis lupus`, `timber wolf`, …) and 51 corpus captions was a good use of
  generation. I reviewed every entry — see below for the one that bit me.
- **Test enumeration.** Given the guard's rule table, proposing a test per rule
  plus the parametrised invalid-payload cases was fast and thorough.
- **Docstring drafting.** I rewrote most of them to say *why* rather than
  *what*, but the first pass was useful scaffolding.

---

## Where AI was wrong, and what I changed

### 1. Ranking ties broke on random UUIDs — reproducibility bug

The first version of `rank_by_similarity` sorted with
`key=lambda pair: (-score, owner_id)` and a comment claiming this made
rankings "stable across runs". It does not. `owner_id` is a random UUID
generated at ingest, so two images with genuinely identical concept vectors
(`coffee-01` and `coffee-04`) would swap places between one seed and the next —
and the eval number would move with them.

I caught this while reading the eval output and noticing a similarity of
exactly `1.000` on a *failing* case. Exact ties are real in a concept-space
embedding; the fix was to break them on a **caller-supplied stable key** (the
filename), not on the row id:

```python
scored.sort(key=lambda pair: (-pair[1], keys.get(pair[0], pair[0])))
```

**Lesson:** a confident comment asserting a property is not evidence of the
property. This one would have shown up as a flaky eval number weeks later.

### 2. `"dark"` as a synonym for the `night` concept

The generated lexicon listed `dark` under the `night` modifier. Reasonable in
isolation — wrong here, because `coffee-01`'s caption is *"A single espresso in
a white cup"* with the attribute `dark` (as in dark roast). The espresso post
therefore ranked `coffee-04` (beans) above `coffee-01` (the actual espresso).

I found it by dumping the extracted concepts per image rather than by staring
at the ranking. Removing `dark` moved top-1 precision from **0.857 to 0.929**.

**Lesson:** a hand-written lexicon needs the same scrutiny as code. "Looks
plausible" is not review. The debugging technique that worked was printing the
*intermediate representation* (extracted concepts), not the final score.

### 3. `extra={"created": ...}` crashed logging — and only sometimes

`ingest_directory` logged its summary dict with `logger.info(..., extra=summary)`.
`summary` contains a key `created`. `logging.Logger.makeRecord` raises
`KeyError: "Attempt to overwrite 'created' in LogRecord"` — `created` is a
reserved LogRecord attribute. So is `filename`, which the image-tagging log
also used.

The nasty part: it only fires when the logger is actually enabled at that
level. Test-ordering decided whether the root logger was at INFO or WARNING, so
the suite failed **three tests intermittently** and I initially misread it as a
seeding race.

The one-line fix would have been renaming two keys. I added
`safe_extra()` in `app/core/logging.py` and routed every `extra=` through it
instead, because the next dynamic dict someone logs will hit this again.

**Lesson:** an intermittent test failure was pointing at a real crash on the
ingestion path, not at a flaky fixture. I nearly added a retry to the test.

### 4. Routes were constructing ORM entities

While writing the EVIDENCE section on layering I ran the grep I was about to
paste as proof — and it returned `3`, not `0`. The route handlers were doing
`repo.add(models.Post(**payload.model_dump()))` and comparing against
`models.VERDICT_ACCEPTED` directly. Layered "in spirit", leaky in fact.

I extracted `app/services/content.py` and `app/services/review_service.py` and
reduced the routes to parse-delegate-present. The grep now genuinely returns
`0`.

**Lesson:** writing the evidence *first* and then running it is a better audit
than reading your own architecture diagram. Two of the four bugs in this list
were found by trying to prove a claim rather than by testing.

### 5. Over-confident threshold justification

My first `tune_thresholds.py` printed a sweep and declared a single "best"
value — which was just the first row of a completely flat plateau from 0.30 to
0.86. The number was meaningless, and the README would have claimed "0.62
maximises accuracy", which is false: so does 0.31.

Rewrote it to report what is actually true: the plateau bounds, the **safe
band** (lowest correct-match similarity vs. highest must-reject similarity), the
max-margin midpoint, and the configured value's margin to each edge. The
README now explains that the plateau is wide *because the tag check already
does the work*, and that 0.62 sits above the midpoint as a deliberate
product decision.

**Lesson:** "the optimiser picked it" is not a defence when the objective is
flat. The interesting finding — the guard's tag rule dominates the similarity
threshold — was hiding inside the bad summary.

---

## Things I chose not to do

- **pgvector.** At 51 images a full scan of a JSON column is microseconds.
  Adding an extension to look sophisticated would have been the wrong call, and
  it would have broken the SQLite path that makes the test suite fast. The
  swap point is documented in the README's limitations.
- **A real broker (Celery/RQ/Redis).** A table-backed queue with
  `FOR UPDATE SKIP LOCKED` is correct for this workload and keeps
  `docker compose up` to three services. Its limits are written down.
- **Relabelling the failing eval case.** `quiet-morning-on-the-beach` picks the
  wave photo over the empty-sand photo because the post body says "the sea is
  flat". I could have edited the post text and reported 1.000. A 100% score on
  a set you tuned by editing is worth nothing, so it stays at 0.929 with the
  failure analysed in the README.
- **Authentication on the review API.** Out of scope for the brief, and a
  half-implemented auth layer is worse than an honestly absent one.
  `X-Tenant-ID` is documented as a scoping header, not a security boundary.

---

## What I would explain in an interview

**Why the wolf gets rejected even though it ranks 7th and scores 0.36.**
Because ranking and refusing are different jobs. On the guard's own test the
wolf is rejected at **0.94** similarity too — `test_wolf_is_rejected_for_a_fox
_post_even_at_high_similarity`. That test is the whole capstone in fifteen
lines: cosine similarity is doing its job correctly and is still not enough,
because "close in meaning" and "the same animal" are different claims.

**Why every failing rule is reported, not just the first.** A reviewer looking
at a refusal needs to know whether the image was the wrong animal, or the right
animal at low confidence, or both. Short-circuiting on the first failure would
save three lines and cost the explanation its value.

**Why failed AI calls are in the cost ledger at $0.** They consumed quota and
latency. A ledger that only records successes hides exactly the incident you
want to see.
