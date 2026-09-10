"""HTTP contract: validation at the boundary, the probes, and the review flow."""

import pytest


# --- validation at the boundary (shared requirement #2) ----------------------
def test_health_reports_the_active_providers(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vision_provider"] == "stub"
    assert body["database"] == "ok"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"slug": "ok-slug", "title": "hi", "body": "too short"},
        {"slug": "Not A Slug", "title": "A title", "body": "long enough body here"},
        {"slug": "ok-slug", "title": "A title"},
        {"slug": "ok-slug", "title": "A title", "body": 42},
    ],
)
def test_bad_post_payloads_are_422_never_500(client, tenant_id, payload):
    response = client.post("/v1/posts", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["errors"]


def test_unknown_job_kind_is_a_422(client, tenant_id):
    response = client.post("/v1/jobs", json={"kind": "make_coffee"})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_out_of_range_query_parameter_is_a_422(client, tenant_id):
    assert client.get("/v1/images?limit=99999").status_code == 422


def test_unknown_post_is_a_404_with_a_useful_message(client, tenant_id):
    response = client.get("/v1/posts/does-not-exist")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_unknown_tenant_is_a_404_not_a_silent_new_tenant(client, tenant_id):
    response = client.get("/v1/posts", headers={"X-Tenant-ID": "nope"})
    assert response.status_code == 404
    assert "unknown tenant" in response.json()["message"]


def test_duplicate_slug_is_a_409(client, tenant_id):
    payload = {
        "slug": "a-post",
        "title": "A post",
        "body": "A body long enough to pass validation.",
    }
    assert client.post("/v1/posts", json=payload).status_code == 201
    response = client.post("/v1/posts", json=payload)
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


def test_every_response_carries_a_request_id(client, tenant_id):
    assert client.get("/health").headers["X-Request-ID"]


# --- tenant isolation (shared requirement #4) --------------------------------
def test_tenants_cannot_see_each_others_posts(client, session, tenant_id):
    from app.repositories.tenants import ensure_tenant

    ensure_tenant(session, "other", name="Other workspace")
    session.commit()

    client.post(
        "/v1/posts",
        json={
            "slug": "tenant-a-post",
            "title": "Tenant A",
            "body": "Body long enough to pass validation.",
        },
    )
    mine = client.get("/v1/posts").json()
    theirs = client.get("/v1/posts", headers={"X-Tenant-ID": "other"}).json()

    assert [p["slug"] for p in mine] == ["tenant-a-post"]
    assert theirs == []


# --- the acceptance probes ---------------------------------------------------
def test_probe1_batch_job_tags_everything_and_flags_the_unsure(seeded_client):
    stats = seeded_client.get("/v1/images/stats").json()
    assert stats["by_status"] == {"tagged": stats["total"]}
    assert stats["flagged_for_review"] >= 1

    flagged = seeded_client.get("/v1/images?needs_review=true").json()
    assert all(image["confidence"] < 0.70 for image in flagged)
    assert all(image["review_reason"] for image in flagged)


def test_probe2_fox_post_ranks_the_fox_first(seeded_client):
    # limit=60 so the whole corpus is ranked and the wolf/dog positions are
    # visible, not just the top slice.
    body = seeded_client.get(
        "/v1/posts/red-foxes-in-deep-snow/images?limit=60"
    ).json()

    assert body["has_confident_match"] is True
    assert body["best_match"]["subject"] == "red fox"
    assert body["best_match"]["rank"] == 1
    assert {s["subject"] for s in body["suggestions"]} == {"red fox"}

    # the wolf and the dog are ranked, but strictly below the fox — and refused
    rank_of = {s["subject"]: s["rank"] for s in body["rejected"]}
    assert "gray wolf" in rank_of and "dog" in rank_of
    assert body["best_match"]["rank"] < rank_of["gray wolf"]
    assert body["best_match"]["rank"] < rank_of["dog"]


def test_probe3_forcing_the_wolf_is_refused_with_an_explanation(seeded_client):
    images = seeded_client.get("/v1/images?category=animal&limit=200").json()
    wolf = next(i for i in images if i["filename"].startswith("gray-wolf-snow-01"))

    response = seeded_client.post(
        f"/v1/posts/red-foxes-in-deep-snow/images/{wolf['id']}/check"
    )
    assert response.status_code == 200
    body = response.json()

    assert body["accepted"] is False
    reason = body["reasons"][0]
    assert reason["code"] == "subject_mismatch"
    assert "expected red fox" in reason["message"]
    assert "detected gray wolf" in reason["message"]
    assert body["signals"]["image_subject_concept"] == "gray wolf"


def test_probe4_post_with_no_good_image_says_so_with_reasons(seeded_client):
    response = seeded_client.get("/v1/posts/restoring-vintage-motorcycles/images")
    assert response.status_code == 200
    assert response.headers["X-Match-Verdict"] == "no-confident-match"

    body = response.json()
    assert body["has_confident_match"] is False
    assert body["best_match"] is None
    assert body["suggestions"] == []
    assert body["no_match"]["message"]
    assert body["no_match"]["rejection_counts"]
    assert body["no_match"]["candidates_considered"] > 0


def test_probe5_eval_endpoint_reports_top1_precision(seeded_client):
    body = seeded_client.post("/v1/eval/run").json()
    assert body["labeled_cases"] >= 10
    assert 0.0 <= body["top1_precision"] <= 1.0
    assert body["top1_precision"] >= 0.8
    assert body["correct_rejection_rate"] == 1.0


def test_probe6_every_ai_call_has_a_cost_entry(seeded_client):
    summary = seeded_client.get("/v1/costs/summary").json()
    assert summary["total_calls"] > 0
    assert summary["total_cost_usd"] > 0
    assert summary["budget_remaining_usd"] < summary["daily_budget_usd"]

    calls = seeded_client.get("/v1/costs?limit=1000").json()
    assert len(calls) == summary["total_calls"]
    assert all(c["subject_ref"] and c["job_id"] for c in calls)
    assert {c["operation"] for c in calls} == {"vision", "embedding"}


# --- the review workflow -----------------------------------------------------
def test_review_approve_reject_and_inspect_why(seeded_client):
    body = seeded_client.get("/v1/posts/red-foxes-in-deep-snow/images").json()
    suggestion_id = body["best_match"]["suggestion_id"]
    assert suggestion_id

    explain = seeded_client.get(f"/v1/suggestions/{suggestion_id}/explain").json()
    assert explain["verdict"] == "accepted"
    assert explain["signals"]["image_subject_concept"] == "red fox"
    assert explain["image_embedding_text"] and explain["post_embedding_text"]
    assert explain["review"] is None

    created = seeded_client.post(
        "/v1/reviews",
        json={
            "suggestion_id": suggestion_id,
            "decision": "approved",
            "reviewer": "editor@example.com",
            "note": "Correct fox, good crop.",
        },
    )
    assert created.status_code == 201

    explain_after = seeded_client.get(
        f"/v1/suggestions/{suggestion_id}/explain"
    ).json()
    assert explain_after["review"]["decision"] == "approved"


def test_explaining_a_rejection_shows_the_reason(seeded_client):
    body = seeded_client.get("/v1/posts/red-foxes-in-deep-snow/images").json()
    wolf = next(s for s in body["rejected"] if s["subject"] == "gray wolf")

    explain = seeded_client.get(
        f"/v1/suggestions/{wolf['suggestion_id']}/explain"
    ).json()
    assert explain["verdict"] == "rejected"
    assert explain["reasons"][0]["code"] == "subject_mismatch"


def test_review_is_idempotent_under_a_replayed_key(seeded_client):
    body = seeded_client.get("/v1/posts/gray-wolves-snowfield/images").json()
    suggestion_id = body["best_match"]["suggestion_id"]
    payload = {
        "suggestion_id": suggestion_id,
        "decision": "rejected",
        "reviewer": "editor@example.com",
    }
    headers = {"Idempotency-Key": "review-key-1"}

    first = seeded_client.post("/v1/reviews", json=payload, headers=headers)
    second = seeded_client.post("/v1/reviews", json=payload, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.headers["Idempotent-Replay"] == "true"
    assert first.json()["id"] == second.json()["id"]

    history = seeded_client.get(
        f"/v1/reviews?suggestion_id={suggestion_id}"
    ).json()
    assert len(history) == 1


def test_reusing_a_key_for_a_different_suggestion_is_a_409(seeded_client):
    suggestions = seeded_client.get("/v1/suggestions?limit=5").json()
    a, b = suggestions[0], suggestions[1]
    headers = {"Idempotency-Key": "shared-key"}

    seeded_client.post(
        "/v1/reviews",
        json={
            "suggestion_id": a["suggestion_id"],
            "decision": "approved",
            "reviewer": "e",
        },
        headers=headers,
    )
    clash = seeded_client.post(
        "/v1/reviews",
        json={
            "suggestion_id": b["suggestion_id"],
            "decision": "approved",
            "reviewer": "e",
        },
        headers=headers,
    )
    assert clash.status_code == 409


def test_invalid_review_decision_is_a_422(seeded_client):
    suggestions = seeded_client.get("/v1/suggestions?limit=1").json()
    response = seeded_client.post(
        "/v1/reviews",
        json={
            "suggestion_id": suggestions[0]["suggestion_id"],
            "decision": "maybe",
            "reviewer": "e",
        },
    )
    assert response.status_code == 422


# --- jobs over HTTP ----------------------------------------------------------
def test_job_submission_is_idempotent(client, tenant_id):
    payload = {"kind": "vision_tagging", "dedupe_key": "nightly-run"}
    first = client.post("/v1/jobs", json=payload)
    second = client.post("/v1/jobs", json=payload)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.headers["Idempotent-Replay"] == "true"
    assert first.json()["id"] == second.json()["id"]


def test_job_progress_is_readable(seeded_client):
    jobs = seeded_client.get("/v1/jobs").json()
    assert jobs
    job = jobs[0]
    assert job["status"] == "succeeded"
    assert job["processed_items"] > 0
    assert job["result"]
