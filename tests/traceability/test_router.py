"""Tests for life_core.traceability.router — /traceability/graph."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.traceability import router as traceability_mod


def _build_app(monkeypatch) -> FastAPI:
    monkeypatch.setattr(
        traceability_mod, "runs_for_deliverable", lambda session, slug: []
    )
    monkeypatch.setattr(
        traceability_mod, "lineage", lambda session, node_ids: []
    )

    def _fake_session():
        yield object()

    app = FastAPI()
    app.dependency_overrides[traceability_mod.get_session] = _fake_session
    app.include_router(traceability_mod.router)
    return app


def test_graph_returns_runs_and_relations_lists(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    response = client.get(
        "/traceability/graph",
        params={"deliverable_slug": "kxkm-batt-16ch"},
    )

    assert response.status_code == 200
    assert response.json() == {"runs": [], "relations": []}


def test_graph_requires_deliverable_slug(monkeypatch):
    app = _build_app(monkeypatch)
    client = TestClient(app)

    response = client.get("/traceability/graph")

    assert response.status_code == 422


def test_graph_serializes_runs_and_relations(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4

    run_id = uuid4()
    rel_id = uuid4()
    artifact_id = uuid4()

    fake_run = SimpleNamespace(
        id=run_id,
        deliverable_slug="kxkm-batt-16ch",
        deliverable_type="hardware",
        role="designer",
        outer_state_at_start="SPEC_READY",
        inner_state="DRAFT",
        verdict=None,
        gate_category=None,
    )
    fake_rel = SimpleNamespace(
        id=rel_id,
        from_id=run_id,
        from_kind="agent_run",
        to_id=artifact_id,
        to_kind="artifact",
        relation_type="derives_from",
    )

    monkeypatch.setattr(
        traceability_mod,
        "runs_for_deliverable",
        lambda session, slug: [fake_run],
    )
    monkeypatch.setattr(
        traceability_mod,
        "lineage",
        lambda session, node_ids: [fake_rel],
    )

    def _fake_session():
        yield object()

    app = FastAPI()
    app.dependency_overrides[traceability_mod.get_session] = _fake_session
    app.include_router(traceability_mod.router)

    response = TestClient(app).get(
        "/traceability/graph",
        params={"deliverable_slug": "kxkm-batt-16ch"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["runs"][0]["id"] == str(run_id)
    assert body["runs"][0]["deliverable_slug"] == "kxkm-batt-16ch"
    assert body["relations"][0]["from_id"] == str(run_id)
    assert body["relations"][0]["to_id"] == str(artifact_id)
    assert body["relations"][0]["relation_type"] == "derives_from"
