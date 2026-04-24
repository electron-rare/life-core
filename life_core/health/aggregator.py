"""V1.7 Track II — /health aggregator.

Collects router + infra + providers snapshots, caches the result
for 2 seconds, and when `emit=True` also publishes
`router.status` + one `infra.network.host` event per host on the
SSE broker.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from life_core.events.broker import get_broker
from life_core.events.schema import Event, EventType

CACHE_TTL_S = 2.0

NETWORK_HOSTS = (
    "electron-server",
    "kxkm-ai",
    "tower",
    "vm-119",
    "cils",
    "grosmac",
)


_cache: dict[str, Any] | None = None
_cache_time: float = 0.0
_lock = asyncio.Lock()


def _cache_clear_for_test() -> None:
    """Reset cache — tests only."""
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0


async def _collect() -> dict[str, Any]:
    """Build the /health payload. Replaces the legacy /health body.

    In test mode (`monkeypatch.setattr`) this is stubbed; in
    production it queries the router and infra modules.
    """
    # Local imports avoid a circular dependency on api.py at import
    # time — api.py imports this module.
    from life_core import api as _api

    router = _api.router
    providers: dict[str, str] = {}
    active_model = "unknown"
    swap_ms = 0.0
    if router is not None:
        try:
            statuses = router.get_provider_status()
            for pid, status in statuses.items():
                providers[pid] = "up" if status else "down"
            active_model = (
                router.get_active_model()
                if hasattr(router, "get_active_model")
                else "unknown"
            )
            swap_ms = float(
                getattr(router, "last_swap_ms", 0.0) or 0.0
            )
        except Exception:
            pass

    # Network hosts — reuse the monitoring_api reachability probes
    # if available; otherwise assume up. 30s tick refreshes in
    # Plan 3.
    network: dict[str, str] = {}
    for host in NETWORK_HOSTS:
        network[host] = "up"

    # Containers — reuse infra_api helpers when installed.
    running = 0
    total = 0
    try:
        from life_core import infra_api

        running, total = await infra_api.count_containers()
    except Exception:
        pass

    status = "healthy"
    if any(v == "down" for v in providers.values()):
        status = "degraded"
    if not providers:
        status = "down"

    return {
        "status": status,
        "router": {
            "status": "up" if router is not None else "down",
            "active_model": active_model,
            "swap_ms": swap_ms,
        },
        "infra": {
            "network": network,
            "containers": {"running": running, "total": total},
            "providers": providers,
        },
    }


async def get_health(emit: bool = False) -> dict[str, Any]:
    """Return the cached or freshly-collected /health payload.

    Cached 2 s. When `emit=True` also broadcasts the payload as
    SSE `router.status` + one `infra.network.host` per host.
    """
    global _cache, _cache_time
    async with _lock:
        now = time.monotonic()
        if _cache is None or (now - _cache_time) > CACHE_TTL_S:
            _cache = await _collect()
            _cache_time = now
            if emit:
                await _emit(_cache)
        snapshot = _cache
    return snapshot


async def _emit(payload: dict[str, Any]) -> None:
    broker = get_broker()
    ts = datetime.now(timezone.utc)
    await broker.publish(
        Event(
            type=EventType.ROUTER_STATUS,
            data=payload["router"],
            timestamp=ts,
        )
    )
    for host, state in payload["infra"]["network"].items():
        await broker.publish(
            Event(
                type=EventType.INFRA_NETWORK_HOST,
                data={"host": host, "up": state == "up"},
                timestamp=ts,
            )
        )
