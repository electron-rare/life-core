"""V1.7 — Keycloak JWT validator for Bucket B (api.saillant.cc)."""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
from authlib.jose import JsonWebKey, jwt
from authlib.jose.errors import JoseError
from fastapi import HTTPException, Request, status

REALM_ISSUER = os.environ.get(
    "KEYCLOAK_ISSUER",
    "https://auth.saillant.cc/realms/life-services",
)
AUDIENCE = os.environ.get("KEYCLOAK_AUDIENCE", "life-core")
ALLOWED_AZP = frozenset({"dolibarr", "meet-backend", "browser-use"})
JWKS_TTL_SECONDS = 300

_jwks_cache: dict[str, Any] = {"keys": None, "fetched_at": 0.0}
_jwks_lock = threading.Lock()


def _fetch_jwks() -> dict[str, Any]:
    url = f"{REALM_ISSUER}/protocol/openid-connect/certs"
    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()
    return response.json()


def _get_jwks() -> dict[str, Any]:
    now = time.time()
    with _jwks_lock:
        if (
            _jwks_cache["keys"] is None
            or now - _jwks_cache["fetched_at"] > JWKS_TTL_SECONDS
        ):
            _jwks_cache["keys"] = _fetch_jwks()
            _jwks_cache["fetched_at"] = now
        return _jwks_cache["keys"]


def validate_keycloak_jwt(request: Request) -> None:
    """Validate Keycloak-issued M2M access token.

    Only runs when caller sets `X-Auth-Mode: bearer` — otherwise
    the life-internal middleware handles the request.
    """
    if request.headers.get("X-Auth-Mode") != "bearer":
        return None

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer",
        )
    token = header[len("Bearer ") :]

    try:
        jwks = _get_jwks()
        key_set = JsonWebKey.import_key_set(jwks)
        claims = jwt.decode(token, key_set)
        claims.validate(now=int(time.time()))
    except JoseError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid jwt: {exc}",
        ) from exc

    aud = claims.get("aud")
    aud_ok = aud == AUDIENCE or (
        isinstance(aud, list) and AUDIENCE in aud
    )
    if not aud_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="wrong audience",
        )

    azp = claims.get("azp")
    if azp not in ALLOWED_AZP:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"azp {azp!r} not allowed",
        )

    request.state.keycloak_azp = azp
    request.state.keycloak_sub = claims.get("sub")
    return None
