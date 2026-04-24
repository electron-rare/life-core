"""V1.7 Track II — direct Langfuse traces proxy.

Reads ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY``. Calls the public API with basic auth
(public=user, secret=password) per Langfuse docs. Returns
``{traces, cursor}`` and emits a ``langfuse.trace`` SSE event per
new trace id, de-duplicated in-process.
"""
from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from life_core.events.broker import get_broker
from life_core.events.schema import Event, EventType

TIMEOUT_S = 5.0
_seen_trace_ids: set[str] = set()


def _auth_header() -> dict[str, str]:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


async def fetch_traces(
    cursor: str | None = None, limit: int = 50
) -> dict[str, Any]:
    """Fetch one page of Langfuse traces and side-emit new ids.

    Returns ``{"traces": [...], "cursor": <next|None>}``. If
    ``LANGFUSE_HOST`` is unset, returns an empty page without
    hitting the network.
    """
    host = os.environ.get("LANGFUSE_HOST", "").rstrip("/")
    if not host:
        return {"traces": [], "cursor": None}
    params: dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.get(
            f"{host}/api/public/traces",
            headers=_auth_header(),
            params=params,
        )
        resp.raise_for_status()
        body = resp.json()
    traces = body.get("data", []) or []
    cursor_next = (body.get("meta") or {}).get("nextCursor")
    await _emit_new(traces)
    return {"traces": traces, "cursor": cursor_next}


async def _emit_new(traces: list[dict[str, Any]]) -> None:
    broker = get_broker()
    ts = datetime.now(timezone.utc)
    for t in traces:
        tid = t.get("id")
        if not tid or tid in _seen_trace_ids:
            continue
        _seen_trace_ids.add(tid)
        await broker.publish(
            Event(
                type=EventType.LANGFUSE_TRACE,
                data={
                    "trace_id": tid,
                    "cost": t.get("totalCost"),
                    "latency": t.get("latency"),
                },
                timestamp=ts,
            )
        )
