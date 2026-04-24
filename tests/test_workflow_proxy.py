"""V1.7 Track II Task 9 — /workflow passthrough to engine."""
from __future__ import annotations

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")
    monkeypatch.setenv(
        "WORKFLOW_ENGINE_URL", "https://engine.saillant.cc"
    )


@respx.mock
def test_workflow_proxy_get_passthrough():
    from life_core.api import app

    respx.get(
        "https://engine.saillant.cc/api/runs?limit=5"
    ).mock(
        return_value=Response(200, json={"runs": [{"id": 1}]})
    )

    client = TestClient(app)
    resp = client.get(
        "/workflow/api/runs?limit=5",
        headers={"Authorization": "Bearer sekret"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"runs": [{"id": 1}]}


@respx.mock
def test_workflow_proxy_post_passthrough():
    from life_core.api import app

    route = respx.post(
        "https://engine.saillant.cc/api/trigger"
    ).mock(return_value=Response(202, json={"accepted": True}))

    client = TestClient(app)
    resp = client.post(
        "/workflow/api/trigger",
        headers={"Authorization": "Bearer sekret"},
        json={"name": "dummy"},
    )
    assert resp.status_code == 202
    assert resp.json() == {"accepted": True}
    assert route.called


@respx.mock
def test_workflow_proxy_forwards_non_2xx():
    from life_core.api import app

    respx.get("https://engine.saillant.cc/api/missing").mock(
        return_value=Response(404, json={"error": "nope"})
    )

    client = TestClient(app)
    resp = client.get(
        "/workflow/api/missing",
        headers={"Authorization": "Bearer sekret"},
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": "nope"}
