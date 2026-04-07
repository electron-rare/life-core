"""Configuration API — providers, platform health, and user preferences."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("life_core.config_api")

router = APIRouter(prefix="/config", tags=["config"])

# Known providers and their env var names
PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": None,
    "vllm": None,
}

PROVIDER_HEALTH_URLS: dict[str, str] = {
    "ollama": os.getenv("OLLAMA_URL", "http://host.docker.internal:11434") + "/api/tags",
    "vllm": (os.getenv("VLLM_BASE_URL", "") or "http://localhost:11434") + "/health",
}

PLATFORM_SERVICES = {
    "redis": os.getenv("REDIS_URL", "redis://redis:6379"),
    "qdrant": os.getenv("QDRANT_URL", "http://qdrant:6333"),
    "ollama": os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
    "vllm": os.getenv("VLLM_BASE_URL", ""),
    "langfuse": os.getenv("LANGFUSE_HOST", ""),
}

REDIS_PROVIDERS_PREFIX = "finefab:config:providers"
REDIS_PREFERENCES_KEY = "finefab:config:preferences"

_redis_client: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis | None:
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        try:
            _redis_client = aioredis.from_url(redis_url, db=0, decode_responses=True)
        except Exception as e:
            logger.warning("Config Redis init failed: %s", e)
            return None
    return _redis_client


def _mask_key(key: str) -> str:
    """Return masked version: first 6 chars + *** + last 4 chars."""
    if not key or len(key) < 10:
        return "***"
    return f"{key[:6]}***{key[-4:]}"


async def _get_provider_key(name: str) -> tuple[str | None, str]:
    """Return (raw_key, source) where source is 'env', 'redis', or 'unconfigured'."""
    # 1. Check env var
    env_var = PROVIDERS.get(name)
    if env_var:
        val = os.getenv(env_var)
        if val:
            return val, "env"

    # 2. Check Redis
    r = await _get_redis()
    if r:
        try:
            val = await r.get(f"{REDIS_PROVIDERS_PREFIX}:{name}")
            if val:
                data = json.loads(val)
                if data.get("api_key"):
                    return data["api_key"], "redis"
        except Exception as e:
            logger.warning("Redis provider key read failed for %s: %s", name, e)

    return None, "unconfigured"


async def _get_provider_meta(name: str) -> dict:
    """Return Redis-stored metadata for a provider (active, priority)."""
    r = await _get_redis()
    if r:
        try:
            val = await r.get(f"{REDIS_PROVIDERS_PREFIX}:{name}")
            if val:
                return json.loads(val)
        except Exception:
            pass
    return {}


# ── Models ────────────────────────────────────────────────────────────────────

class ProviderInfo(BaseModel):
    name: str
    source: str          # env | redis | unconfigured
    masked_key: str | None
    active: bool
    priority: int


class ProviderUpdate(BaseModel):
    api_key: str | None = None
    active: bool | None = None
    priority: int | None = None


class ProviderTestResult(BaseModel):
    name: str
    ok: bool
    latency_ms: float | None = None
    error: str | None = None


class ServiceHealth(BaseModel):
    name: str
    ok: bool
    url: str
    memory: str | None = None
    error: str | None = None


class PlatformHealth(BaseModel):
    services: list[ServiceHealth]


class Preferences(BaseModel):
    default_model: str = ""
    rag_enabled: bool = False
    language: str = "FR"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers() -> list[ProviderInfo]:
    """List all known providers with masked keys, status, and priority."""
    result: list[ProviderInfo] = []
    for name in PROVIDERS:
        raw_key, source = await _get_provider_key(name)
        meta = await _get_provider_meta(name)
        result.append(ProviderInfo(
            name=name,
            source=source,
            masked_key=_mask_key(raw_key) if raw_key else None,
            active=meta.get("active", raw_key is not None),
            priority=meta.get("priority", list(PROVIDERS).index(name)),
        ))
    return result


@router.put("/providers/{name}", response_model=ProviderInfo)
async def update_provider(name: str, body: ProviderUpdate) -> ProviderInfo:
    """Update a provider's API key, active flag, or priority."""
    if name not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {name}")

    r = await _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    # Load existing data
    existing: dict[str, Any] = {}
    try:
        val = await r.get(f"{REDIS_PROVIDERS_PREFIX}:{name}")
        if val:
            existing = json.loads(val)
    except Exception:
        pass

    if body.api_key is not None:
        existing["api_key"] = body.api_key
    if body.active is not None:
        existing["active"] = body.active
    if body.priority is not None:
        existing["priority"] = body.priority

    try:
        await r.set(f"{REDIS_PROVIDERS_PREFIX}:{name}", json.dumps(existing))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis write failed: {e}") from e

    raw_key, source = await _get_provider_key(name)
    return ProviderInfo(
        name=name,
        source=source,
        masked_key=_mask_key(raw_key) if raw_key else None,
        active=existing.get("active", raw_key is not None),
        priority=existing.get("priority", list(PROVIDERS).index(name)),
    )


@router.post("/providers/{name}/test", response_model=ProviderTestResult)
async def test_provider(name: str) -> ProviderTestResult:
    """Test connectivity for a provider."""
    if name not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {name}")

    raw_key, source = await _get_provider_key(name)

    # For API-key providers, check if key exists at all
    if name in ("anthropic", "openai", "mistral", "groq", "google"):
        if not raw_key:
            return ProviderTestResult(name=name, ok=False, error="No API key configured")

    # Ping health endpoint for local providers
    url = PROVIDER_HEALTH_URLS.get(name)
    if url:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                latency_ms = (time.perf_counter() - start) * 1000
                ok = resp.status_code < 400
                return ProviderTestResult(
                    name=name,
                    ok=ok,
                    latency_ms=round(latency_ms, 1),
                    error=None if ok else f"HTTP {resp.status_code}",
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderTestResult(name=name, ok=False, latency_ms=round(latency_ms, 1), error=str(e))

    # For cloud providers with a key, do a minimal API ping
    test_urls: dict[str, tuple[str, dict]] = {
        "anthropic": ("https://api.anthropic.com/v1/models", {"x-api-key": raw_key or "", "anthropic-version": "2023-06-01"}),
        "openai": ("https://api.openai.com/v1/models", {"Authorization": f"Bearer {raw_key or ''}"}),
        "mistral": ("https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {raw_key or ''}"}),
        "groq": ("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {raw_key or ''}"}),
        "google": ("https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": raw_key or ""}),
    }

    if name in test_urls:
        url, headers = test_urls[name]
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, headers=headers)
                latency_ms = (time.perf_counter() - start) * 1000
                ok = resp.status_code < 400
                return ProviderTestResult(
                    name=name,
                    ok=ok,
                    latency_ms=round(latency_ms, 1),
                    error=None if ok else f"HTTP {resp.status_code}",
                )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderTestResult(name=name, ok=False, latency_ms=round(latency_ms, 1), error=str(e))

    return ProviderTestResult(name=name, ok=False, error="No test method available")


@router.get("/platform", response_model=PlatformHealth)
async def platform_health() -> PlatformHealth:
    """Aggregate health check for Redis, Qdrant, Ollama, vLLM, Langfuse."""
    services: list[ServiceHealth] = []

    # Redis — direct ping
    redis_url = PLATFORM_SERVICES["redis"]
    try:
        r = await _get_redis()
        if r:
            await r.ping()
            services.append(ServiceHealth(name="redis", ok=True, url=redis_url))
        else:
            services.append(ServiceHealth(name="redis", ok=False, url=redis_url, error="Client init failed"))
    except Exception as e:
        services.append(ServiceHealth(name="redis", ok=False, url=redis_url, error=str(e)))

    # Qdrant — GET /collections
    qdrant_url = PLATFORM_SERVICES["qdrant"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{qdrant_url}/collections")
            ok = resp.status_code < 400
            services.append(ServiceHealth(name="qdrant", ok=ok, url=qdrant_url,
                                          error=None if ok else f"HTTP {resp.status_code}"))
    except Exception as e:
        services.append(ServiceHealth(name="qdrant", ok=False, url=qdrant_url, error=str(e)))

    # Ollama — GET /api/tags
    ollama_url = PLATFORM_SERVICES["ollama"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            ok = resp.status_code < 400
            memory: str | None = None
            if ok:
                data = resp.json()
                count = len(data.get("models", []))
                memory = f"{count} model(s)"
            services.append(ServiceHealth(name="ollama", ok=ok, url=ollama_url,
                                          memory=memory,
                                          error=None if ok else f"HTTP {resp.status_code}"))
    except Exception as e:
        services.append(ServiceHealth(name="ollama", ok=False, url=ollama_url, error=str(e)))

    # vLLM — optional
    vllm_url = PLATFORM_SERVICES["vllm"]
    if vllm_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vllm_url}/health")
                ok = resp.status_code < 400
                services.append(ServiceHealth(name="vllm", ok=ok, url=vllm_url,
                                              error=None if ok else f"HTTP {resp.status_code}"))
        except Exception as e:
            services.append(ServiceHealth(name="vllm", ok=False, url=vllm_url, error=str(e)))
    else:
        services.append(ServiceHealth(name="vllm", ok=False, url="", error="VLLM_BASE_URL not set"))

    # Langfuse — optional
    langfuse_url = PLATFORM_SERVICES["langfuse"]
    if langfuse_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{langfuse_url}/api/public/health")
                ok = resp.status_code < 400
                services.append(ServiceHealth(name="langfuse", ok=ok, url=langfuse_url,
                                              error=None if ok else f"HTTP {resp.status_code}"))
        except Exception as e:
            services.append(ServiceHealth(name="langfuse", ok=False, url=langfuse_url, error=str(e)))
    else:
        services.append(ServiceHealth(name="langfuse", ok=False, url="", error="LANGFUSE_HOST not set"))

    return PlatformHealth(services=services)


@router.get("/preferences", response_model=Preferences)
async def get_preferences() -> Preferences:
    """Get user preferences from Redis."""
    r = await _get_redis()
    if r:
        try:
            val = await r.get(REDIS_PREFERENCES_KEY)
            if val:
                return Preferences.model_validate(json.loads(val))
        except Exception as e:
            logger.warning("Preferences read failed: %s", e)
    return Preferences()


@router.put("/preferences", response_model=Preferences)
async def save_preferences(body: Preferences) -> Preferences:
    """Save user preferences to Redis."""
    r = await _get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    try:
        await r.set(REDIS_PREFERENCES_KEY, body.model_dump_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis write failed: {e}") from e
    return body
