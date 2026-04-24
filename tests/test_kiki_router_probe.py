"""V1.7 Track II Task 6 — Studio kiki-router deep probe.

Validated during the V1.7 brainstorm 2026-04-23 (option B): the
`/providers` response is augmented with a `kiki_router` key that
reports per-model reachability (35 niche + 7 meta LoRA aliases),
the current swap latency in ms, and the currently active LoRA
identifier. This block is cached separately at 15 s because Studio
state changes faster than the 30 s `/providers` envelope.
"""
from __future__ import annotations

import asyncio

import pytest
import respx
from httpx import Response


def test_kiki_router_probe_parses_status_payload():
    from life_core.providers import kiki_router_probe as mod

    payload = {
        "active_lora": "meta-sota-v4",
        "last_swap_ms": 842.5,
        "models": [
            {"alias": "kiki-medical-a3b", "status": "up"},
            {"alias": "kiki-legal-a3b", "status": "up"},
            {"alias": "meta-sota-v4", "status": "up"},
        ],
    }
    parsed = mod._parse_status(payload)
    assert parsed["host"] == "studio"
    assert parsed["active_lora"] == "meta-sota-v4"
    assert parsed["swap_ms"] == 842.5
    assert len(parsed["models"]) == 3
    assert parsed["models"][0] == {
        "alias": "kiki-medical-a3b", "status": "up"
    }


@respx.mock
def test_kiki_router_probe_http_roundtrip_mocked():
    from life_core.providers import kiki_router_probe as mod

    respx.get("http://studio:9200/status").mock(
        return_value=Response(
            200,
            json={
                "active_lora": "meta-sota-v4",
                "last_swap_ms": 512.0,
                "models": [
                    {"alias": a, "status": "up"}
                    for a in (
                        "kiki-medical-a3b",
                        "kiki-legal-a3b",
                        "meta-sota-v4",
                    )
                ],
            },
        )
    )

    mod._cache_clear_for_test()
    result = asyncio.run(mod.probe_once())
    assert result["host"] == "studio"
    assert result["active_lora"] == "meta-sota-v4"
    assert result["swap_ms"] == 512.0
    assert {m["alias"] for m in result["models"]} == {
        "kiki-medical-a3b", "kiki-legal-a3b", "meta-sota-v4",
    }


@respx.mock
def test_kiki_router_probe_handles_timeout():
    from life_core.providers import kiki_router_probe as mod

    respx.get("http://studio:9200/status").mock(
        side_effect=asyncio.TimeoutError
    )

    mod._cache_clear_for_test()
    result = asyncio.run(mod.probe_once())
    assert result["host"] == "studio"
    assert result["active_lora"] is None
    assert result["swap_ms"] is None
    assert result["models"] == []


def test_deep_probe_cache_is_15s(monkeypatch):
    from life_core.providers import kiki_router_probe as mod

    count = 0

    async def fake_fetch():
        nonlocal count
        count += 1
        return {
            "host": "studio",
            "active_lora": "meta-sota-v4",
            "swap_ms": 100.0,
            "models": [],
        }

    monkeypatch.setattr(mod, "_fetch_status", fake_fetch)
    mod._cache_clear_for_test()

    async def run():
        await mod.probe_once()
        await mod.probe_once()
        await mod.probe_once()

    asyncio.run(run())
    # Cache coalesces: only one fetch within 15 s window.
    assert count == 1
    assert mod.CACHE_TTL_S == 15.0


def test_probe_respects_env_url_override(monkeypatch):
    """KIKI_ROUTER_STATUS_URL env overrides the default studio URL."""
    monkeypatch.setenv(
        "KIKI_ROUTER_STATUS_URL",
        "http://other-host:9999/status",
    )
    # Re-import so module picks up env.
    import importlib

    from life_core.providers import kiki_router_probe as mod

    mod = importlib.reload(mod)
    assert mod.STATUS_URL == "http://other-host:9999/status"


def test_parse_status_tolerates_missing_fields():
    """Partial payloads must not blow up the parser."""
    from life_core.providers import kiki_router_probe as mod

    parsed = mod._parse_status({})
    assert parsed["host"] == "studio"
    assert parsed["active_lora"] is None
    assert parsed["swap_ms"] is None
    assert parsed["models"] == []

    # Entries without alias are skipped; missing status defaults to
    # "unknown" so the parser never raises on partial data.
    parsed = mod._parse_status(
        {"models": [{"status": "up"}, {"alias": "ok"}]}
    )
    assert parsed["models"] == [{"alias": "ok", "status": "unknown"}]


def test_providers_response_embeds_kiki_router(monkeypatch):
    """The /providers payload is augmented with kiki_router block."""
    from fastapi.testclient import TestClient

    from life_core import api as _api
    from life_core.providers import kiki_router_probe as kmod
    from life_core.providers import registry

    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")
    monkeypatch.setenv("KIKI_FULL_BASE_URL", "http://kxkm-ai:8000")
    monkeypatch.setenv("KIKI_FULL_MODELS", "qwen-14b-awq-kxkm")

    async def fake_probe(entry):
        return {
            "id": entry["id"],
            "status": "up",
            "model_count": len(entry["models"]),
        }

    async def fake_deep_probe():
        return {
            "host": "studio",
            "active_lora": "meta-sota-v4",
            "swap_ms": 842.5,
            "models": [
                {"alias": "kiki-medical-a3b", "status": "up"},
            ],
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
    assert "kiki_router" in body
    assert body["kiki_router"]["host"] == "studio"
    assert body["kiki_router"]["active_lora"] == "meta-sota-v4"
    assert body["kiki_router"]["swap_ms"] == 842.5
    assert body["kiki_router"]["models"][0]["alias"] == "kiki-medical-a3b"
