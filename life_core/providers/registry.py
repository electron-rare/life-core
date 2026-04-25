"""V1.7 Track II — provider registry + reachability probes.

Seed the registry from env (``KIKI_FULL_BASE_URL``,
``KIKI_FULL_MODELS``, ``VLLM_BASE_URL``, ``VLLM_MODELS``, etc.) and
probe each in parallel with a 2 s timeout. Results are cached for
30 s. On every refresh a ``router.status`` summary is pushed onto
the SSE broker so the cockpit sidebar stays live.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

CACHE_TTL_S = 30.0
PROBE_TIMEOUT_S = 2.0


@dataclass
class ProviderEntry:
    id: str
    base_url: str
    models: list[str] = field(default_factory=list)
    probe_path: str = "/v1/models"


def list_entries() -> list[dict[str, Any]]:
    """Build the provider list from env. Ids match spec naming."""
    entries: list[dict[str, Any]] = []

    kxkm_base = os.getenv("KIKI_FULL_BASE_URL") or os.getenv(
        "VLLM_BASE_URL"
    )
    kxkm_models = [
        m.strip()
        for m in os.getenv(
            "KIKI_FULL_MODELS", os.getenv("VLLM_MODELS", "")
        ).split(",")
        if m.strip()
    ]
    if kxkm_base and kxkm_models:
        entries.append(
            {
                "id": "kxkm-vllm",
                "base_url": kxkm_base,
                "models": kxkm_models,
                "probe_path": "/v1/models",
            }
        )

    local_base = os.getenv("LOCAL_LLM_BASE_URL")
    local_models = [
        m.strip()
        for m in os.getenv("LOCAL_LLM_MODELS", "").split(",")
        if m.strip()
    ]
    if local_base and local_models:
        entries.append(
            {
                "id": "studio-router",
                "base_url": local_base,
                "models": local_models,
                "probe_path": "/v1/models",
            }
        )

    ollama_base = os.getenv("OLLAMA_URL")
    if ollama_base:
        entries.append(
            {
                "id": "ollama-cils",
                "base_url": ollama_base,
                "models": [],  # Ollama lists at runtime.
                "probe_path": "/api/tags",
            }
        )

    return entries


_cache: dict[str, Any] | None = None
_cache_time: float = 0.0
_lock = asyncio.Lock()


def _cache_clear_for_test() -> None:
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0


async def _http_head(url: str) -> int:
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as client:
        resp = await client.get(url)
        return resp.status_code


async def _probe_one(entry: dict[str, Any]) -> dict[str, Any]:
    url = entry["base_url"].rstrip("/") + entry.get(
        "probe_path", "/v1/models"
    )
    status = "down"
    try:
        code = await asyncio.wait_for(
            _http_head(url), timeout=PROBE_TIMEOUT_S
        )
        if 200 <= code < 500:
            status = "up"
    except Exception:
        status = "down"
    return {
        "id": entry["id"],
        "status": status,
        "model_count": len(entry.get("models", [])),
    }


async def get_providers() -> dict[str, Any]:
    global _cache, _cache_time
    async with _lock:
        now = time.monotonic()
        if _cache is None or (now - _cache_time) > CACHE_TTL_S:
            entries = list_entries()
            probes = await asyncio.gather(
                *(_probe_one(e) for e in entries),
                return_exceptions=False,
            )
            _cache = {"providers": list(probes)}
            # V1.7 Track II — augment with Studio kiki-router deep
            # probe. Separate 15 s TTL inside probe_once().
            from life_core.providers import kiki_router_probe

            _cache["kiki_router"] = await kiki_router_probe.probe_once()
            _cache_time = now
            await _emit(_cache)
        return _cache


async def _emit(payload: dict[str, Any]) -> None:
    """Publish a router.status summary on the broker."""
    from datetime import datetime, timezone

    from life_core.events.broker import get_broker
    from life_core.events.schema import Event, EventType

    broker = get_broker()
    up_count = sum(
        1 for p in payload["providers"] if p["status"] == "up"
    )
    kiki_router = payload.get("kiki_router") or {}
    await broker.publish(
        Event(
            type=EventType.ROUTER_STATUS,
            data={
                "providers_up": up_count,
                "providers_total": len(payload["providers"]),
                "kiki_router_active_lora": kiki_router.get(
                    "active_lora"
                ),
                "kiki_router_swap_ms": kiki_router.get("swap_ms"),
            },
            timestamp=datetime.now(timezone.utc),
        )
    )


# V1.8 Wave B axes 1+6 — MCP catalog surfaced to cockpit + shim.
_MCP_CATALOG: list[dict] = [
    {
        "name": "datasheet",
        "transport": "sse",
        "url": "http://datasheet-mcp:8021/sse",
        "http_url": "http://datasheet-mcp:8022",
        "bearer_env": "DATASHEET_BEARER",
        "capabilities": ["upload", "search", "detail"],
        "since": "v1.8",
    },
    {
        "name": "docstore",
        "transport": "sse",
        "url": "http://docstore-mcp:8020/sse",
        "http_url": "http://docstore-mcp:8020",
        "capabilities": ["search"],
        "since": "v1.6",
    },
    {
        "name": "cad",
        "transport": "sse",
        "url": "http://cad-mcp:8022/sse",
        "http_url": "http://cad-mcp:8022",
        "capabilities": [
            "get_schematic",
            "get_bom",
            "get_netlist",
            "get_drc_results",
            "read_partial_sch",
            "add_component",
            "add_wire",
            "place_footprint",
            "route_track",
            "apply_design_block",
        ],
        "since": "v1.8",
    },
]


def get_mcp_catalog() -> list[dict]:
    """V1.8 Wave B axes 1+6 — return the MCP catalog.

    Cockpit /infra panel and axis-10 RAG consumers both read this.
    List is static for V1.8; V1.9 sources it from Infisical.
    """
    return list(_MCP_CATALOG)
