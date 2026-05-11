"""Configuration API — providers, platform health, and user preferences."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


def resolve_llm_config() -> dict:
    """Map LIFE_CORE_LLM_BACKEND env to LiteLLM model_list config.

    Returns a dict with:
      primary_model   — canonical model name to use for LLM calls
      fallback_models — ordered list of fallback model names (may be empty)

    Backend values:
      auto           (default) — claude-runner primary, qwen-14b fallback
      claude-runner  — claude-runner only, no fallback
      qwen           — qwen-14b only, no fallback
    """
    backend = os.getenv("LIFE_CORE_LLM_BACKEND", "auto").strip().lower()
    if backend == "claude-runner":
        return {"primary_model": "claude-runner-sonnet-4-7", "fallback_models": []}
    if backend == "qwen":
        return {"primary_model": "qwen-14b-awq-kxkm", "fallback_models": []}
    # auto: primary claude-runner, fallback qwen
    return {
        "primary_model": "claude-runner-sonnet-4-7",
        "fallback_models": ["qwen-14b-awq-kxkm"],
    }


def default_catalog_entry(model_id: str) -> dict:
    """Compose a minimal catalog entry for a model the hand-curated
    YAML does not know about. Prefix-based dispatch: kiki-meta-*,
    kiki-niche-*, anthropic/*, openai/gpt-*, openai/qwen-*, etc."""
    if model_id.startswith("kiki-meta-"):
        intent = model_id.removeprefix("kiki-meta-")
        return {
            "id": model_id,
            "name": f"Kiki meta ({intent})",
            "provider": "kiki-router",
            "domain": intent,
            "description": (
                f"Multi-adapter routing via kiki-router for {intent} meta "
                f"intent. Runs on Studio MLX."
            ),
            "size": "19 GB base + 35 LoRAs",
            "location": "Studio M3 Ultra",
            "context_window": "262K tokens",
        }
    if model_id.startswith("ailiance-"):
        # eu-kiki gateway aliases (electron-server:9300). One alias maps to
        # one backend worker; the gateway hides routing.
        _AILIANCE_META: dict[str, dict] = {
            "ailiance-qwen": {
                "name": "Qwen3-Next 80B (FR/multilingue, lourd)",
                "size": "Q4_K_M GGUF (80B params)",
                "location": "KXKM-AI RTX 4090",
                "context_window": "262K tokens",
                "description": (
                    "Qwen3-Next-80B-A3B-Instruct Q4_K_M served by llama.cpp. "
                    "Heavy reasoning, multilingue."
                ),
            },
            "ailiance-granite": {
                "name": "Granite-4.1 30B",
                "size": "Q4_K_M GGUF",
                "location": "KXKM-AI RTX 4090",
                "context_window": "131K tokens",
                "description": "IBM Granite-4.1 30B served by llama.cpp.",
            },
            "ailiance-gemma": {
                "name": "Gemma 3 4B (rapide)",
                "size": "GGUF",
                "location": "Tower P2000",
                "context_window": "131K tokens",
                "description": "gemma-3-4b-it GGUF served by llama.cpp.",
            },
            "ailiance-eurollm": {
                "name": "EuroLLM 22B (FR)",
                "size": "MLX 4-bit",
                "location": "MacStudio M3 Ultra",
                "context_window": "32K tokens",
                "description": (
                    "EuroLLM 22B fine-tuned FR/multilingue. Préféré pour "
                    "chat-fr, traduction-tech, redaction-multilingue, "
                    "localisation-doc."
                ),
            },
            "ailiance-mistral-medium": {
                "name": "Mistral-Medium 3.5 128B",
                "size": "Q8 MLX (~128B params)",
                "location": "MacStudio M3 Ultra",
                "context_window": "32K tokens",
                # studio:9301 hosts Mistral-Medium 128B Q8 (launchd
                # cc.ailiance.mistral.plist). Apertus 70B was never deployed.
                "description": (
                    "Mistral-Medium 3.5 128B MLX Q8 servi par le worker "
                    "studio:9301 (launchd cc.ailiance.mistral.plist). "
                    "Général / raisonnement lourd."
                ),
            },
        }
        if model_id in _AILIANCE_META:
            meta = _AILIANCE_META[model_id]
            return {
                "id": model_id,
                "name": meta["name"],
                "provider": "eu-kiki",
                "domain": "general",
                "description": meta["description"],
                "size": meta["size"],
                "location": meta["location"],
                "context_window": meta["context_window"],
            }
        # MLX bundle on macM1 :8502 (Ministral, Gemma2/4, Qwen2.5, Llama-3.2…)
        suffix = model_id.removeprefix("ailiance-")
        return {
            "id": model_id,
            "name": f"MLX bundle ({suffix})",
            "provider": "eu-kiki",
            "domain": "general",
            "description": (
                "Alias eu-kiki vers le bundle MLX (macM1 :8502). 14 modèles "
                "MLX 4-bit disponibles via mlx_lm.server."
            ),
            "size": "MLX 4-bit",
            "location": "macM1",
            "context_window": "32K tokens",
        }
    if model_id.startswith("kiki-niche-"):
        niche = model_id.removeprefix("kiki-niche-")
        return {
            "id": model_id,
            "name": f"Kiki niche ({niche})",
            "provider": "kiki-router",
            "domain": niche,
            "description": (
                f"Qwen3.6-35B-A3B + v4-sota LoRA fine-tuned for {niche} "
                f"domain. Routed via kiki-router on Studio."
            ),
            "size": "19 GB base + 1 LoRA",
            "location": "Studio M3 Ultra",
            "context_window": "262K tokens",
        }
    if model_id.startswith("anthropic/"):
        return {
            "id": model_id,
            "name": model_id.split("/", 1)[1],
            "provider": "anthropic",
            "domain": "general",
            "description": "Cloud-hosted Anthropic Claude model.",
            "size": "cloud",
            "location": "Anthropic API",
            "context_window": "200K tokens",
        }
    if model_id.startswith("groq/"):
        return {
            "id": model_id,
            "name": model_id.split("/", 1)[1],
            "provider": "groq",
            "domain": "general",
            "description": "Groq-hosted model, sub-second TTFT.",
            "size": "cloud",
            "location": "Groq API",
            "context_window": "8K tokens",
        }
    if model_id.startswith("openai/gpt-"):
        return {
            "id": model_id,
            "name": model_id.split("/", 1)[1],
            "provider": "openai",
            "domain": "general",
            "description": "Cloud-hosted OpenAI GPT model.",
            "size": "cloud",
            "location": "OpenAI API",
            "context_window": "128K tokens",
        }
    if model_id.startswith("openai/qwen") or model_id.startswith("openai/mascarade"):
        return {
            "id": model_id,
            "name": model_id.split("/", 1)[1],
            "provider": "vllm",
            "domain": "general",
            "description": "llama-server hosted on KXKM-AI RTX 4090.",
            "size": "Q4_K_M",
            "location": "KXKM-AI",
            "context_window": "128K tokens",
        }
    if model_id.startswith("tei/"):
        return {
            "id": model_id,
            "name": model_id.split("/", 1)[1],
            "provider": "tei",
            "domain": "embedding",
            "description": "text-embeddings-inference server.",
            "size": "local",
            "location": "electron-server",
            "context_window": "8K tokens",
        }
    # Unknown prefix — return a minimal stub so the UI still has something
    # to render rather than hiding the model entirely.
    return {
        "id": model_id,
        "name": model_id,
        "provider": "unknown",
        "domain": "general",
        "description": (
            "No metadata available. Add an entry to "
            "config/models_catalog.yaml to improve this."
        ),
        "size": "unknown",
        "location": "unknown",
        "context_window": "unknown",
    }

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


def _load_ui_features() -> dict[str, bool]:
    """Read F4L_UI_FEATURE_* env vars to build runtime UI flags.

    Each flag defaults to true when the env var is absent — so the
    sidebar renders every section unless explicitly disabled by ops.
    """
    known = [
        "dashboard", "projects", "chat", "search", "providers", "rag",
        "traces", "infra", "monitoring", "governance", "schematic",
        "config", "goose", "datasheets", "workflow", "sse",
    ]
    flags: dict[str, bool] = {}
    for key in known:
        env_name = f"F4L_UI_FEATURE_{key.upper()}"
        raw = os.environ.get(env_name, "true").strip().lower()
        flags[key] = raw in ("1", "true", "yes", "on")
    return flags

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
    ui_features: dict[str, bool] = {}


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

    return PlatformHealth(services=services, ui_features=_load_ui_features())


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
