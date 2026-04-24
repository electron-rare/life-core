"""V1.7 Track II Task 5 — /health aggregator + SSE side-emit."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _bearer(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")


def test_health_shape_matches_spec():
    from life_core.api import app

    client = TestClient(app)
    resp = client.get(
        "/health", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("healthy", "degraded", "down")
    assert set(body["router"].keys()) >= {
        "status",
        "active_model",
        "swap_ms",
    }
    assert set(body["infra"].keys()) >= {
        "network",
        "containers",
        "providers",
    }
    assert set(body["infra"]["network"].keys()) == {
        "electron-server",
        "kxkm-ai",
        "tower",
        "vm-119",
        "cils",
        "grosmac",
    }


def test_health_is_cached_2s(monkeypatch):
    from life_core.health import aggregator

    call_count = 0

    async def fake_collect():
        nonlocal call_count
        call_count += 1
        return {
            "status": "healthy",
            "router": {
                "status": "up",
                "active_model": "qwen-14b-awq-kxkm",
                "swap_ms": 3.6,
            },
            "infra": {
                "network": {k: "up" for k in aggregator.NETWORK_HOSTS},
                "containers": {"running": 92, "total": 93},
                "providers": {"kxkm-vllm": "up"},
            },
        }

    monkeypatch.setattr(aggregator, "_collect", fake_collect)
    aggregator._cache_clear_for_test()

    async def run():
        await aggregator.get_health()
        await aggregator.get_health()
        await aggregator.get_health()
        return call_count

    assert asyncio.run(run()) == 1


@pytest.mark.asyncio
async def test_health_emits_sse_events(monkeypatch):
    from life_core.events.broker import get_broker
    from life_core.events.schema import EventType
    from life_core.health import aggregator

    async def fake_collect():
        return {
            "status": "healthy",
            "router": {
                "status": "up",
                "active_model": "qwen-14b-awq-kxkm",
                "swap_ms": 3.6,
            },
            "infra": {
                "network": {k: "up" for k in aggregator.NETWORK_HOSTS},
                "containers": {"running": 92, "total": 93},
                "providers": {"kxkm-vllm": "up"},
            },
        }

    monkeypatch.setattr(aggregator, "_collect", fake_collect)
    aggregator._cache_clear_for_test()
    broker = get_broker()
    q = broker.subscribe()
    try:
        await aggregator.get_health(emit=True)
        seen = set()
        for _ in range(7):
            ev = await asyncio.wait_for(q.get(), timeout=0.5)
            seen.add(ev.type)
            if (
                EventType.ROUTER_STATUS in seen
                and EventType.INFRA_NETWORK_HOST in seen
            ):
                break
        assert EventType.ROUTER_STATUS in seen
        assert EventType.INFRA_NETWORK_HOST in seen
    finally:
        broker.unsubscribe(q)
