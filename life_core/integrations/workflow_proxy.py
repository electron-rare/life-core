"""V1.7 Track II — GET/POST proxy to engine.saillant.cc.

Minimal pass-through: copies body, forwards the caller's
Authorization header verbatim (F4L bearer downstream), and
returns the downstream status + JSON body.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import Request

TIMEOUT_S = 10.0


def _base_url() -> str:
    return os.environ.get(
        "WORKFLOW_ENGINE_URL", "https://engine.saillant.cc"
    ).rstrip("/")


async def proxy(request: Request, subpath: str) -> tuple[int, Any]:
    """Forward a request to the workflow engine and return
    (status_code, parsed_body).

    - Preserves HTTP method, query params, and raw body.
    - Forwards the caller's ``Authorization`` header verbatim.
    - Falls back to ``{"raw": <text>}`` when the downstream body is
      not valid JSON.
    """
    url = f"{_base_url()}/{subpath.lstrip('/')}"
    params = dict(request.query_params)
    headers: dict[str, str] = {}
    if (auth := request.headers.get("Authorization")) is not None:
        headers["Authorization"] = auth
    body = await request.body()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            params=params,
            content=body if body else None,
        )
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw": resp.text}
