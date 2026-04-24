"""Router tests for ``life_core.agents.router`` (T1.8d)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.agents.router import get_session, router


def _make_app(session_factory) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = session_factory
    return app


def test_run_unknown_role_returns_422():
    app = _make_app(lambda: MagicMock())
    client = TestClient(app)
    r = client.post(
        "/agents/wrong/run",
        json={
            "deliverable_slug": "s",
            "deliverable_type": "spec",
            "outer_state": "spec",
            "compliance_profile": "prototype",
            "upstream_artifacts": [],
            "context": {},
            "hitl_mode": "sync",
        },
    )
    assert r.status_code == 422


def test_run_invalid_payload_returns_422():
    app = _make_app(lambda: MagicMock())
    client = TestClient(app)
    r = client.post("/agents/spec/run", json={"deliverable_slug": "x"})
    assert r.status_code == 422


def test_decide_unknown_role_returns_422():
    app = _make_app(lambda: MagicMock())
    client = TestClient(app)
    r = client.post(
        f"/agents/bogus/decide/{uuid4()}?decision=approve",
    )
    assert r.status_code == 422


def test_decide_unknown_decision_returns_422():
    app = _make_app(lambda: MagicMock())
    client = TestClient(app)
    r = client.post(
        f"/agents/spec/decide/{uuid4()}?decision=boom",
    )
    assert r.status_code == 422


def test_decide_missing_row_returns_404():
    session = MagicMock()
    session.get.return_value = None
    app = _make_app(lambda: session)
    client = TestClient(app)
    r = client.post(
        f"/agents/spec/decide/{uuid4()}?decision=approve",
    )
    assert r.status_code == 404


def test_decide_approve_updates_row():
    session = MagicMock()
    row = MagicMock()
    row.inner_state = "REVIEW"
    session.get.return_value = row
    app = _make_app(lambda: session)
    client = TestClient(app)
    r = client.post(
        f"/agents/spec/decide/{uuid4()}?decision=approve",
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {"ok": True, "inner_state": "APPROVED"}
    assert row.inner_state == "APPROVED"
    session.flush.assert_called_once()


def test_decide_reprompt_maps_to_draft():
    session = MagicMock()
    row = MagicMock()
    row.inner_state = "REVIEW"
    session.get.return_value = row
    app = _make_app(lambda: session)
    client = TestClient(app)
    r = client.post(
        f"/agents/spec/decide/{uuid4()}?decision=reprompt",
    )
    assert r.status_code == 200
    assert r.json()["inner_state"] == "DRAFT"


def test_get_run_missing_returns_404():
    session = MagicMock()
    session.get.return_value = None
    app = _make_app(lambda: session)
    client = TestClient(app)
    r = client.get(f"/agents/runs/{uuid4()}")
    assert r.status_code == 404


def test_get_run_returns_row_snapshot():
    session = MagicMock()
    run_id = uuid4()
    row = MagicMock()
    row.id = run_id
    row.deliverable_slug = "s"
    row.role = "spec"
    row.inner_state = "APPROVED"
    row.verdict = "GateSpecPass"
    row.started_at = datetime(2026, 4, 24, 10, 0, tzinfo=timezone.utc)
    session.get.return_value = row
    app = _make_app(lambda: session)
    client = TestClient(app)
    r = client.get(f"/agents/runs/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["deliverable_slug"] == "s"
    assert body["inner_state"] == "APPROVED"
    assert body["verdict"] == "GateSpecPass"
    assert body["started_at"].startswith("2026-04-24T10:00")
