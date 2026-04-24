"""V1.7 Track II Task 6 — /providers endpoint + registry."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _bearer(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")
    # Seed at least one provider for the endpoint round-trip.
    monkeypatch.setenv("KIKI_FULL_BASE_URL", "http://kxkm-ai:8000")
    monkeypatch.setenv("KIKI_FULL_MODELS", "qwen-14b-awq-kxkm")


def test_providers_endpoint_returns_registry(monkeypatch):
    from life_core import api as _api
    from life_core.providers import kiki_router_probe as kmod
    from life_core.providers import registry

    async def fake_probe(entry):
        return {
            "id": entry["id"],
            "status": "up",
            "model_count": len(entry["models"]),
        }

    async def fake_deep_probe():
        return {
            "host": "studio",
            "active_lora": None,
            "swap_ms": None,
            "models": [],
        }

    monkeypatch.setattr(registry, "_probe_one", fake_probe)
    monkeypatch.setattr(kmod, "probe_once", fake_deep_probe)
    registry._cache_clear_for_test()
    kmod._cache_clear_for_test()

    client = TestClient(_api.app)
    resp = client.get(
        "/providers", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "providers" in body
    assert {p["id"] for p in body["providers"]} >= {"kxkm-vllm"}
    assert all(
        set(p.keys()) == {"id", "status", "model_count"}
        for p in body["providers"]
    )


def test_probe_timeout_marks_down(monkeypatch):
    from life_core.providers import registry

    async def slow_probe(*args, **kwargs):
        await asyncio.sleep(5)
        return None

    monkeypatch.setattr(registry, "_http_head", slow_probe)
    entry = {
        "id": "kxkm-vllm",
        "base_url": "http://kxkm-ai:8000",
        "models": ["qwen-14b-awq-kxkm"],
    }
    result = asyncio.run(registry._probe_one(entry))
    assert result == {
        "id": "kxkm-vllm",
        "status": "down",
        "model_count": 1,
    }


def test_registry_is_cached_30s(monkeypatch):
    from life_core.providers import kiki_router_probe as kmod
    from life_core.providers import registry

    count = 0

    async def fake_probe(entry):
        nonlocal count
        count += 1
        return {
            "id": entry["id"],
            "status": "up",
            "model_count": 1,
        }

    async def fake_deep_probe():
        return {
            "host": "studio",
            "active_lora": None,
            "swap_ms": None,
            "models": [],
        }

    monkeypatch.setattr(registry, "_probe_one", fake_probe)
    monkeypatch.setattr(kmod, "probe_once", fake_deep_probe)
    registry._cache_clear_for_test()
    kmod._cache_clear_for_test()

    async def run():
        await registry.get_providers()
        await registry.get_providers()
        await registry.get_providers()

    asyncio.run(run())
    # Only one round of probes per provider.
    assert count == len(registry.list_entries())
