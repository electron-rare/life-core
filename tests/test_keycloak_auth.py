"""V1.7 Task 8 — tests for Keycloak JWT validator middleware."""
from __future__ import annotations

import time

import pytest
from authlib.jose import jwt, JsonWebKey
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from life_core.middleware import keycloak_auth
from life_core.middleware.keycloak_auth import validate_keycloak_jwt


@pytest.fixture
def rsa_keypair():
    key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
    return key


@pytest.fixture
def app(monkeypatch, rsa_keypair):
    jwks_public = {"keys": [rsa_keypair.as_dict(is_private=False)]}

    def fake_jwks():
        return jwks_public

    monkeypatch.setattr(keycloak_auth, "_fetch_jwks", fake_jwks)

    app = FastAPI()

    @app.get(
        "/v1/b/ping",
        dependencies=[Depends(validate_keycloak_jwt)],
    )
    def ping():
        return {"ok": True}

    return app


def _token(rsa_keypair, **overrides):
    now = int(time.time())
    claims = {
        "iss": "https://auth.saillant.cc/realms/life-services",
        "aud": "life-core",
        "azp": "dolibarr",
        "exp": now + 300,
        "iat": now,
    }
    claims.update(overrides)
    header = {"alg": "RS256", "kid": rsa_keypair.kid}
    return jwt.encode(header, claims, rsa_keypair).decode()


def test_missing_header_returns_401(app):
    client = TestClient(app)
    resp = client.get(
        "/v1/b/ping",
        headers={"X-Auth-Mode": "bearer"},
    )
    assert resp.status_code == 401


def test_valid_jwt_for_dolibarr_passes(app, rsa_keypair):
    token = _token(rsa_keypair, azp="dolibarr")
    client = TestClient(app)
    resp = client.get(
        "/v1/b/ping",
        headers={
            "X-Auth-Mode": "bearer",
            "Authorization": f"Bearer {token}",
        },
    )
    assert resp.status_code == 200


def test_wrong_audience_returns_401(app, rsa_keypair):
    token = _token(rsa_keypair, aud="other-service")
    client = TestClient(app)
    resp = client.get(
        "/v1/b/ping",
        headers={
            "X-Auth-Mode": "bearer",
            "Authorization": f"Bearer {token}",
        },
    )
    assert resp.status_code == 401


def test_unknown_azp_returns_401(app, rsa_keypair):
    token = _token(rsa_keypair, azp="rogue-client")
    client = TestClient(app)
    resp = client.get(
        "/v1/b/ping",
        headers={
            "X-Auth-Mode": "bearer",
            "Authorization": f"Bearer {token}",
        },
    )
    assert resp.status_code == 401


def test_expired_token_returns_401(app, rsa_keypair):
    token = _token(rsa_keypair, exp=int(time.time()) - 10)
    client = TestClient(app)
    resp = client.get(
        "/v1/b/ping",
        headers={
            "X-Auth-Mode": "bearer",
            "Authorization": f"Bearer {token}",
        },
    )
    assert resp.status_code == 401
