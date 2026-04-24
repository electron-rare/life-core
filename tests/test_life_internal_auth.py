"""V1.7 Task 7 — tests for LIFE_INTERNAL_BEARER middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from life_core.middleware.life_internal_auth import (
    validate_life_internal_bearer,
)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "secret-bearer-xyz")
    app = FastAPI()

    @app.get("/v1/protected", dependencies=[Depends(validate_life_internal_bearer)])
    def protected():
        return {"ok": True}

    return app


def test_missing_auth_returns_401(app):
    client = TestClient(app)
    resp = client.get("/v1/protected")
    assert resp.status_code == 401


def test_wrong_bearer_returns_401(app):
    client = TestClient(app)
    resp = client.get(
        "/v1/protected",
        headers={"Authorization": "Bearer wrong-value"},
    )
    assert resp.status_code == 401


def test_correct_bearer_returns_200(app):
    client = TestClient(app)
    resp = client.get(
        "/v1/protected",
        headers={"Authorization": "Bearer secret-bearer-xyz"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_auth_mode_bearer_skips_internal_check(app):
    """When X-Auth-Mode: bearer is set, caller is Bucket B (JWT),
    so the internal-bearer check should not apply. The dependency
    still raises 401 (Bucket A gate) unless the JWT middleware
    runs first; here we only assert the header is respected by
    validate_life_internal_bearer — it returns None without
    raising.
    """
    client = TestClient(app)
    resp = client.get(
        "/v1/protected",
        headers={
            "X-Auth-Mode": "bearer",
            "Authorization": "Bearer irrelevant",
        },
    )
    assert resp.status_code == 200
