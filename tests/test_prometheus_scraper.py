"""V1.7 Task 5 — unit tests for the 7-host Prometheus scraper.

The scraper polls /metrics on the 7 known hosts at a fixed 30 s
interval (per spec §5.7) and publishes a single
`infra.network.host` Event per host per tick via the broker
defined in Plan 2 Task 2.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock

import pytest


def test_default_targets_has_seven_hosts():
    from life_core.monitoring.prometheus_scraper import DEFAULT_TARGETS

    names = {t.host for t in DEFAULT_TARGETS}
    assert names == {
        "tower",
        "kxkm-ai",
        "electron-server",
        "vm-119",
        "cils",
        "grosmac",
        "studio",
    }


def test_scrape_interval_is_30_seconds():
    from life_core.monitoring.prometheus_scraper import SCRAPE_INTERVAL_SECONDS

    # Pinned by spec §5.7 — infra.network.host triggers on 30 s tick.
    assert SCRAPE_INTERVAL_SECONDS == 30


@pytest.mark.asyncio
async def test_scrape_host_returns_up_true_on_200(monkeypatch):
    from life_core.monitoring import prometheus_scraper as mod

    async def fake_get(self, url, timeout):
        class R:
            status_code = 200
            text = "node_cpu_seconds_total 1\n"
        return R()

    monkeypatch.setattr(mod.httpx.AsyncClient, "get", fake_get)
    client = mod.httpx.AsyncClient()
    host, up, latency_ms = await mod.scrape_host(
        client, mod.Target("tower", "http://tower:9100/metrics")
    )
    assert host == "tower"
    assert up is True
    assert latency_ms >= 0


@pytest.mark.asyncio
async def test_scrape_host_returns_up_false_on_timeout(monkeypatch):
    from life_core.monitoring import prometheus_scraper as mod

    async def fake_get(self, url, timeout):
        raise mod.httpx.ConnectTimeout("simulated")

    monkeypatch.setattr(mod.httpx.AsyncClient, "get", fake_get)
    client = mod.httpx.AsyncClient()
    host, up, latency_ms = await mod.scrape_host(
        client, mod.Target("vm-119", "http://192.168.0.119:9100/metrics")
    )
    assert host == "vm-119"
    assert up is False
    assert latency_ms is None


@pytest.mark.asyncio
async def test_scrape_once_publishes_one_event_per_host(monkeypatch):
    from life_core.events.broker import EventBroker
    from life_core.events.schema import EventType
    from life_core.monitoring import prometheus_scraper as mod

    async def fake_scrape_host(client, target):
        return (target.host, True, 7.5)

    monkeypatch.setattr(mod, "scrape_host", fake_scrape_host)

    broker = EventBroker()
    q = broker.subscribe()
    await mod.scrape_once(broker, targets=mod.DEFAULT_TARGETS)

    received: List = []
    while not q.empty():
        received.append(await q.get())

    assert len(received) == 7
    for ev in received:
        assert ev.type is EventType.INFRA_NETWORK_HOST
        assert set(ev.data.keys()) == {"host", "up", "latency_ms"}
        assert ev.data["up"] is True
        assert ev.data["latency_ms"] == 7.5
        assert isinstance(ev.timestamp, datetime)
        assert ev.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_scraper_task_cancels_cleanly(monkeypatch):
    from life_core.events.broker import EventBroker
    from life_core.monitoring import prometheus_scraper as mod

    monkeypatch.setattr(mod, "SCRAPE_INTERVAL_SECONDS", 0.05)

    scraped = []

    async def fake_scrape_once(broker, targets=None):
        scraped.append(datetime.now(timezone.utc))

    monkeypatch.setattr(mod, "scrape_once", fake_scrape_once)

    broker = EventBroker()
    task = asyncio.create_task(mod.run_scraper(broker))
    await asyncio.sleep(0.17)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # At least 2 ticks fired within the ~170ms window.
    assert len(scraped) >= 2
