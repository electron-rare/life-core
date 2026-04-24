"""V1.7 Track II — Studio kiki-router deep probe.

kiki-router on Studio :9200 serves Qwen3.6-35B-A3B v4-SOTA +
35 niche LoRAs + 7 meta LoRAs. The public shim probes kiki-router
via its ``/status`` endpoint (fallback ``/health``) to surface:

- per-model (alias) reachability
- the currently active LoRA identifier
- the last swap latency in ms

The result is cached 15 s independently of the 30 s ``/providers``
envelope because Studio state (LoRA swaps) changes faster than the
provider-list refresh rate, and the cockpit benefits from a tighter
heartbeat for the kiki-router card.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

# Spec §5.7 — deep-probe TTL. Pinned in a test.
CACHE_TTL_S: float = 15.0
PROBE_TIMEOUT_S: float = 2.0

# Studio kiki-router health URL. Override via env for non-prod.
STATUS_URL: str = os.getenv(
    "KIKI_ROUTER_STATUS_URL", "http://studio:9200/status"
)

_cache: dict[str, Any] | None = None
_cache_time: float = 0.0
_lock = asyncio.Lock()


def _cache_clear_for_test() -> None:
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0


def _empty_result() -> dict[str, Any]:
    return {
        "host": "studio",
        "active_lora": None,
        "swap_ms": None,
        "models": [],
    }


def _parse_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise the kiki-router /status JSON into our shape."""
    models_in = payload.get("models") or []
    models = [
        {
            "alias": m.get("alias", ""),
            "status": m.get("status", "unknown"),
        }
        for m in models_in
        if m.get("alias")
    ]
    return {
        "host": "studio",
        "active_lora": payload.get("active_lora"),
        "swap_ms": payload.get("last_swap_ms"),
        "models": models,
    }


async def _fetch_status() -> dict[str, Any]:
    """Issue one HTTP GET to kiki-router and parse the body."""
    try:
        async with httpx.AsyncClient(
            timeout=PROBE_TIMEOUT_S
        ) as client:
            resp = await client.get(STATUS_URL)
            if 200 <= resp.status_code < 300:
                return _parse_status(resp.json())
    except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
        pass
    return _empty_result()


async def probe_once() -> dict[str, Any]:
    """Return the current deep-probe result, cached CACHE_TTL_S seconds."""
    global _cache, _cache_time
    async with _lock:
        now = time.monotonic()
        if _cache is None or (now - _cache_time) > CACHE_TTL_S:
            _cache = await _fetch_status()
            _cache_time = now
        return _cache
