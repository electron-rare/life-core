"""SSE /events endpoint — push snapshot of health/stats/goose.

Consolidates the three polling endpoints (/health, /stats, /goose/stats)
into a single Server-Sent Events stream, reducing dashboard traffic from
~90 req/min to a single long-lived connection with one snapshot every
F4L_SSE_INTERVAL seconds (default 3.0).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Awaitable, TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

events_router = APIRouter()

T = TypeVar("T")


def _get_interval() -> float:
    """Read the SSE interval at call time so tests can override it."""
    return float(os.environ.get("F4L_SSE_INTERVAL", "3.0"))


def _get_max_duration() -> float:
    """Read the max SSE connection lifetime at call time (seconds)."""
    return float(os.environ.get("F4L_SSE_MAX_DURATION", "3600"))


def _get_sub_call_timeout() -> float:
    """Per sub-call timeout used while building the snapshot (seconds)."""
    return float(os.environ.get("F4L_SSE_SUBCALL_TIMEOUT", "2.0"))


SSE_INTERVAL_SECONDS = _get_interval()


async def _safe_call(
    awaitable: Awaitable[T],
    default: T,
    label: str,
    timeout: float | None = None,
) -> T:
    """Await *awaitable* with a timeout and return *default* on failure.

    Used inside ``_snapshot()`` so one slow or broken sub-service does not
    stall the whole SSE stream.
    """
    timeout_s = timeout if timeout is not None else _get_sub_call_timeout()
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("snapshot sub-call %s timed out after %.2fs", label, timeout_s)
        return default
    except Exception as exc:  # noqa: BLE001 — isolation point
        logger.warning("snapshot sub-call %s failed: %s", label, exc)
        return default


async def _snapshot() -> dict:
    """Collect the current platform state as a single JSON-serialisable dict.

    Imports life_core.api lazily to avoid a circular import at module load
    time — events_api is itself registered from api.py.
    """
    # Local import avoids cycle — api.py imports this module.
    from life_core.api import router, cache, chat_service  # noqa: WPS433

    providers: list[str] = []
    router_status: dict[str, bool] = {}
    if router is not None:
        try:
            providers = router.list_available_providers()
        except Exception:
            providers = []
        try:
            router_status = router.get_provider_status()
        except Exception:
            router_status = {}

    chat_stats: dict = {}
    if chat_service is not None:
        try:
            chat_stats = chat_service.get_stats()
        except Exception:
            chat_stats = {}

    # Goose stats — call the endpoint coroutine directly; fall back to
    # a zero-filled stub if it raises or times out (goose daemon down, etc.).
    goose_default: dict[str, Any] = {
        "active_sessions": 0,
        "total_prompts": 0,
        "recipes_available": 0,
    }
    try:
        from life_core.goose_api import goose_stats as _goose_stats_endpoint
        goose_stats = await _safe_call(
            _goose_stats_endpoint(),
            goose_default,
            "goose_stats",
        )
    except Exception:
        goose_stats = goose_default

    status = "ok" if providers and all(
        router_status.values() if router_status else [True]
    ) else "degraded"

    return {
        "health": {
            "status": status,
            "providers": providers,
            "cache_available": cache is not None,
            "router_status": router_status,
        },
        "stats": {
            "chat_service": chat_stats,
            "router": {"status": router_status},
        },
        "goose": goose_stats,
    }


async def _event_generator(request: Request):
    """Async generator yielding SSE-formatted snapshots until disconnect.

    Emits an ``event: reconnect`` frame and closes the loop once the
    connection has been open for more than ``F4L_SSE_MAX_DURATION`` seconds
    (default 3600). The client is expected to reopen the stream.
    """
    start = time.monotonic()
    max_duration_s = _get_max_duration()
    while True:
        if await request.is_disconnected():
            logger.debug("SSE client disconnected")
            break
        if time.monotonic() - start > max_duration_s:
            logger.info(
                "SSE generator reached max duration (%.1fs), closing",
                max_duration_s,
            )
            yield "event: reconnect\ndata: {}\n\n"
            break
        try:
            snapshot = await _snapshot()
            payload = json.dumps(snapshot)
            yield f"event: snapshot\ndata: {payload}\n\n"
        except Exception as exc:
            logger.warning("SSE snapshot failed: %s", exc)
            error_payload = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error_payload}\n\n"
        await asyncio.sleep(_get_interval())


@events_router.get("/events")
async def events(request: Request):
    """Server-Sent Events stream of platform snapshots."""
    return StreamingResponse(
        _event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
