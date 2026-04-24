"""V1.7 — shared bearer gate for Bucket A (life-internal network)."""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status


def validate_life_internal_bearer(request: Request) -> None:
    """Enforce the shared Bucket A bearer on /v1/* requests.

    Bucket A containers (suite-numerique, meet-transcriber, grist)
    share the bearer defined by env `LIFE_INTERNAL_BEARER`.

    When a caller sets `X-Auth-Mode: bearer`, it declares itself
    Bucket B — JWT validation happens in `keycloak_auth.py`
    (see Task 8) and this dependency short-circuits to None.

    Uses `hmac.compare_digest` for constant-time comparison.
    """
    if request.headers.get("X-Auth-Mode") == "bearer":
        return None

    expected = os.environ.get("LIFE_INTERNAL_BEARER", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="life-core not configured with LIFE_INTERNAL_BEARER",
        )

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer",
        )
    token = header[len("Bearer ") :]

    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid life-internal bearer",
        )
    return None
