"""V1.7 Track II Task 12 — /datasheets stub (wired in V1.8)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _bearer(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")


def test_datasheets_stub_exact_shape():
    from life_core.api import app

    client = TestClient(app)
    resp = client.get(
        "/datasheets",
        headers={"Authorization": "Bearer sekret"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "items": [],
        "message": "not wired — see V1.8 roadmap",
    }
