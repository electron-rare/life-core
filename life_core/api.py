"""API FastAPI pour life-core."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import json

import httpx
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from life_core.events.broker import get_broker
from life_core.monitoring.prometheus_scraper import install_startup_hook
from life_core.middleware.life_internal_auth import (
    validate_life_internal_bearer,
)
from life_core.middleware.keycloak_auth import validate_keycloak_jwt
from pydantic import BaseModel, Field

V1_AUTH_DEPS = [
    Depends(validate_life_internal_bearer),
    Depends(validate_keycloak_jwt),
]

from life_core.cache import MultiTierCache
from life_core.rag import RAGPipeline
from life_core.rag.api import rag_router, set_rag_pipeline
from life_core.infra_api import infra_router
from life_core.monitoring_api import monitoring_router
from life_core.ws_alerts import ws_router as ws_alerts_router
from life_core.traces_api import traces_router
from life_core.stats_api import stats_router
from life_core.logs_api import logs_router
from life_core.conversations_api import conversations_router, set_redis
from life_core.models_api import models_router, set_models_router
from life_core.audit_api import audit_router
from life_core.goose_api import router as goose_router
from life_core.projects.router import router as projects_router, team_router, set_redis as set_projects_redis
from life_core.config_api import router as config_router
from life_core.router import LiteLLMProvider, Router
from life_core.services import BrowserService, ChatService
from life_core.services.browser import (
    BrowserDependencyMissingError,
    BrowserRemoteRunnerError,
    BrowserServiceError,
)
from life_core.langfuse_tracing import flush_langfuse, init_langfuse
from life_core.telemetry import init_telemetry
from life_core.docstore_client import augment_with_docstore

logger = logging.getLogger("life_core.api")

# W3C traceparent propagation
try:
    from opentelemetry.context import attach, detach
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    _propagator = TraceContextTextMapPropagator()
    _has_otel_propagation = True
except ImportError:
    _has_otel_propagation = False

# Global instances
router: Router | None = None
cache: MultiTierCache | None = None
rag: RAGPipeline | None = None
chat_service: ChatService | None = None
browser_service: BrowserService | None = None

# Session Redis client (DB 3 — distinct from cache DB 0 and nc-rag-indexer DB 2)
MAX_CONTEXT_MESSAGES = 20
_session_redis: aioredis.Redis | None = None


async def _get_session_redis() -> aioredis.Redis | None:
    """Lazy-init async Redis client on DB 3 for session storage."""
    global _session_redis
    if _session_redis is None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        try:
            _session_redis = aioredis.from_url(redis_url, db=3)
        except Exception as e:
            logger.warning(f"Session Redis init failed: {e}")
            return None
    return _session_redis


async def _load_session(session_id: str) -> list[dict]:
    """Load conversation history from Redis. Returns [] on any failure."""
    try:
        r = await _get_session_redis()
        if r is None:
            return []
        data = await r.get(f"rag:session:{session_id}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Session load failed for {session_id}: {e}")
    return []


async def _save_session(session_id: str, messages: list[dict]) -> None:
    """Persist conversation history to Redis with 24h TTL. Silent on failure."""
    try:
        r = await _get_session_redis()
        if r is None:
            return
        trimmed = messages[-50:]
        await r.set(f"rag:session:{session_id}", json.dumps(trimmed), ex=86400)
    except Exception as e:
        logger.warning(f"Session save failed for {session_id}: {e}")


def _trim_messages(messages: list[dict], max_messages: int = MAX_CONTEXT_MESSAGES) -> list[dict]:
    """Keep system prompt + last N non-system messages to fit context window."""
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    return system + non_system[-max_messages:]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'app."""
    global router, cache, rag, chat_service, browser_service
    
    # Startup
    logger.info("Initializing life-core API")

    # Initialize OpenTelemetry (no-op if OTEL_EXPORTER_OTLP_ENDPOINT not set)
    init_telemetry()
    init_langfuse()

    # Initialiser le routeur
    router = Router()
    
    # --- LiteLLM unified provider ---
    DEFAULT_MODELS = {
        "OPENAI_API_KEY": ["openai/gpt-4o", "openai/gpt-4o-mini"],
        "ANTHROPIC_API_KEY": ["anthropic/claude-sonnet-4-20250514"],
        "MISTRAL_API_KEY": ["mistral/mistral-large-latest"],
        "GROQ_API_KEY": ["groq/llama3-70b-8192"],
        "GOOGLE_API_KEY": ["gemini/gemini-2.0-flash"],
    }

    models: list[str] = []
    for env_var, default_models in DEFAULT_MODELS.items():
        if os.getenv(env_var):
            models += default_models

    # --- vLLM provider (KXKM-AI GPU) ---
    vllm_base = os.getenv("VLLM_BASE_URL")
    vllm_models_str = os.getenv("VLLM_MODELS", "")
    vllm_models: set[str] = set()
    if vllm_base and vllm_models_str:
        vllm_models = {m.strip() for m in vllm_models_str.split(",")}
        models += list(vllm_models)
        logger.info("vLLM models registered: %s via %s", vllm_models, vllm_base)

    # --- Local LLM provider (llama.cpp on Tower GPU P2000) ---
    local_llm_base = os.getenv("LOCAL_LLM_URL")
    local_llm_models_str = os.getenv("LOCAL_LLM_MODELS", "")
    local_llm_models: set[str] = set()
    if local_llm_base and local_llm_models_str:
        local_llm_models = {m.strip() for m in local_llm_models_str.split(",")}
        models += list(local_llm_models)
        logger.info("Local LLM models registered: %s via %s", local_llm_models, local_llm_base)

    # --- kiki-router (micro-kiki full_pipeline_server via 2-hop tunnel) ---
    kiki_full_base = os.getenv("KIKI_FULL_BASE_URL")
    kiki_full_models_str = os.getenv("KIKI_FULL_MODELS", "")
    kiki_full_models: set[str] = set()
    if kiki_full_base and kiki_full_models_str:
        kiki_full_models = {m.strip() for m in kiki_full_models_str.split(",") if m.strip()}
        models += list(kiki_full_models)
        logger.info("Kiki router models registered: %d via %s", len(kiki_full_models), kiki_full_base)

    if override := os.getenv("LITELLM_MODELS"):
        models = [m.strip() for m in override.split(",")]

    if models:
        provider = LiteLLMProvider(
            models=models,
            vllm_api_base=vllm_base,
            vllm_models=vllm_models,
            local_llm_api_base=local_llm_base,
            local_llm_models=local_llm_models,
            kiki_full_base_url=kiki_full_base,
            kiki_full_models=kiki_full_models,
        )
        router.register_provider(provider, is_primary=True)
        logger.info("LiteLLM provider registered with %d models: %s", len(models), models)
    else:
        logger.warning("No LLM API keys found — no provider registered")

    # --- LiteLLM callbacks → Langfuse ---
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        import litellm
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
        logger.info("LiteLLM Langfuse callbacks enabled")

    # Initialiser le cache
    redis_url = os.getenv("REDIS_URL")
    cache = MultiTierCache(redis_url=redis_url)
    
    # Initialiser le RAG (optionnel)
    try:
        qdrant_url = os.environ.get("QDRANT_URL")
        rag = RAGPipeline(qdrant_url=qdrant_url)
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}")
        rag = None
    set_rag_pipeline(rag)

    # Créer le service de chat
    chat_service = ChatService(router=router, cache=cache, rag=rag)
    browser_service = BrowserService()

    # Wire Redis to conversations and projects cache
    if cache and hasattr(cache, '_redis') and cache._redis:
        set_redis(cache._redis)
        set_projects_redis(cache._redis)
    else:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis as redis_lib
                r = redis_lib.from_url(redis_url)
                set_redis(r)
                set_projects_redis(r)
            except Exception:
                pass

    # Wire router to models_api so /models/catalog can backfill from /models
    set_models_router(router)

    providers = router.list_available_providers()
    logger.info(f"life-core initialized with {len(providers)} providers: {providers}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down life-core API")
    set_models_router(None)
    flush_langfuse()


# Créer l'app
app = FastAPI(
    title="life-core API",
    description="Life-core LLM router and RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# V1.7 Task 5 — attach the 7-host Prometheus scraper as a startup task.
install_startup_hook(app)

app.include_router(rag_router)
app.include_router(infra_router)
app.include_router(monitoring_router)
app.include_router(ws_alerts_router)
app.include_router(traces_router)  # V1.8 axis 5 — /traces/inner for HITL cockpit feed
app.include_router(stats_router)
app.include_router(logs_router)
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(audit_router)
app.include_router(goose_router)
app.include_router(projects_router)
app.include_router(team_router)
app.include_router(config_router)

from life_core.agents.router import router as agents_router
app.include_router(agents_router)

from life_core.evaluations.router import router as evaluations_router
app.include_router(evaluations_router)

from life_core.traceability.router import router as traceability_router
app.include_router(traceability_router)

from life_core.providers_api import providers_router
app.include_router(providers_router)

from life_core.events_api import events_router
app.include_router(events_router)

try:
    from life_core.mcp_server import mcp as mcp_server
    app.mount("/mcp", mcp_server.streamable_http_app())
    logger.info("MCP server mounted at /mcp")
except ImportError:
    logger.warning("MCP SDK not installed — /mcp endpoint disabled")
except Exception as e:
    logger.warning("MCP mount failed: %s", e)

@app.middleware("http")
async def propagate_trace_context(request, call_next):
    """Extract W3C traceparent from incoming request headers and attach to OTEL context."""
    if _has_otel_propagation:
        carrier = dict(request.headers)
        ctx = _propagator.extract(carrier)
        token = attach(ctx)
        try:
            response = await call_next(request)
            return response
        finally:
            detach(token)
    return await call_next(request)


# CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry instrumentations
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HttpxClientInstrumentor
    
    FastAPIInstrumentor().instrument_app(app)
    HttpxClientInstrumentor().instrument()
    logger.info("OpenTelemetry auto-instrumentation for FastAPI/httpx enabled")
except ImportError:
    logger.debug("OpenTelemetry instrumentation packages not available (no-op)")


# Models
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://host.docker.internal:8889")


async def _web_search(query: str, top_k: int = 3) -> list[dict]:
    """Fetch web results from SearXNG. Returns empty list on failure."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r["title"], "url": r["url"], "content": r.get("content", "")}
                for r in data.get("results", [])[:top_k]
            ]
    except Exception as e:
        logger.warning(f"SearXNG search failed: {e}")
        return []


def _format_web_results(results: list[dict]) -> str:
    """Format web results for injection into system prompt."""
    parts = []
    for r in results:
        parts.append(f"- {r['title']}\n  {r['url']}\n  {r['content'][:300]}")
    return "\n".join(parts)


DEFAULT_CHAT_MODEL = "openai/qwen-14b-awq-kxkm"

LEGACY_MODEL_ALIASES = {
    "auto": "openai/qwen-14b-awq-kxkm",
    "openai/qwen-14b-awq": "openai/qwen-14b-awq-kxkm",
    "qwen-14b-awq": "openai/qwen-14b-awq-kxkm",
}


def resolve_model_alias(model_id: str) -> str:
    """Resolve a legacy or bare model id to its canonical -kxkm form.

    Returns the input unchanged if no alias matches. Used by the
    OpenAI-compat shim to accept pre-V1.6 client configurations.
    """
    return LEGACY_MODEL_ALIASES.get(model_id, model_id)


class ChatRequest(BaseModel):
    """Requête de chat."""
    messages: list[dict[str, str]]
    model: str = DEFAULT_CHAT_MODEL
    provider: str | None = None
    use_rag: bool = False
    web_search: bool = True
    session_id: str | None = None
    # Strip <think>...</think> CoT from the kiki-router response. Default
    # False: reasoning-heavy models keep their CoT. Set True for short
    # UI-bound output. Forwarded to micro-kiki full_pipeline_server via
    # LiteLLM per-call body; ignored by non-kiki providers.
    strip_thinking: bool = False


class ChatResponse(BaseModel):
    """Réponse de chat."""

    class Usage(BaseModel):
        """Usage de tokens du provider."""

        input_tokens: int = 0
        output_tokens: int = 0

    content: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    trace_id: str = ""


class ModelsResponse(BaseModel):
    """Réponse des modèles disponibles."""
    models: list[str]


class ScrapeRequest(BaseModel):
    """Requête de scraping via navigateur."""

    url: str = Field(..., min_length=1)
    selector: str | None = None
    timeout_ms: int = Field(default=15000, ge=1, le=120000)


class ScrapeResponse(BaseModel):
    """Réponse de scraping via navigateur."""

    url: str
    title: str
    content: str


# Routes
@app.get("/health")
async def health():
    """V1.7 Track II — aggregated health. Cached 2s.

    Public (no auth) so Docker healthcheck + Traefik liveness
    probes can reach it without credentials. Emits router.status
    + infra.network.host SSE events on each cache refresh so the
    cockpit stays live.
    """
    from life_core.health.aggregator import get_health

    return await get_health(emit=True)


@app.get("/providers", dependencies=V1_AUTH_DEPS)
async def providers_endpoint():
    """V1.7 Track II Task 6 — provider list + reachability probes.

    Probes each configured provider (KIKI_FULL_*, VLLM_*, OLLAMA_URL)
    in parallel with a 2s timeout; cached 30s. Augmented with a
    ``kiki_router`` deep probe of Studio :9200 (cached 15s). Emits a
    router.status SSE event on each refresh.
    """
    from life_core.providers.registry import get_providers

    return await get_providers()


@app.get("/config", dependencies=V1_AUTH_DEPS)
async def config_endpoint():
    """V1.7 Track II Task 8 — read-only config exposure.

    Surfaces allowlisted env vars, the full model list (same
    source as /providers), and the 7-host network map from the
    Prometheus scraper DEFAULT_TARGETS. Any env name matching
    ``*_KEY | *_SECRET | *_TOKEN | *_PASSWORD`` is hard-blocked
    regardless of allowlist. See
    ``life_core.integrations.config_exposure`` for details.
    """
    from life_core.integrations.config_exposure import collect

    return collect()


# V1.7 Track II Task 12 — /datasheets stub.
# Full wiring (digikey/lcsc/element14/mouser) deferred to V1.8
# per docs/superpowers/plans/2026-04-23-v1.7-track-ii-cockpit.md
# Section 5.4 follow-up.
@app.get("/datasheets")
async def datasheets_stub(
    _bearer: None = Depends(validate_life_internal_bearer),
):
    """V1.7 Track II — stub. Full wiring (digikey/lcsc/element14/
    mouser) deferred to V1.8 per spec Section 5.4 follow-up.
    """
    return {
        "items": [],
        "message": "not wired — see V1.8 roadmap",
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """Lister tous les modèles disponibles."""
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")
    
    models: set[str] = set()
    for provider_id in router.list_available_providers():
        try:
            provider = router.providers[provider_id]
            provider_models = await provider.list_models()
            models.update(provider_models)
        except Exception as e:
            logger.warning(f"Failed to list models for {provider_id}: {e}")

    return ModelsResponse(models=sorted(list(models)))


@app.get("/web-search")
async def web_search(q: str, top_k: int = 5):
    """Search the web via SearXNG."""
    results = await _web_search(q, top_k=top_k)
    return {"query": q, "results": results, "count": len(results)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Envoyer un message au chat."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    try:
        # Load session history from Redis if session_id provided
        history: list[dict] = []
        if request.session_id:
            history = await _load_session(request.session_id)

        # Merge history + current messages (avoid duplicate last user message)
        if history:
            last_history_content = history[-1].get("content", "") if history else ""
            first_request_content = request.messages[0].get("content", "") if request.messages else ""
            if last_history_content == first_request_content:
                merged_messages = history
            else:
                merged_messages = history + list(request.messages)
        else:
            merged_messages = list(request.messages)

        # Context-aware RAG: use last 3 user messages as search query
        augmented_messages = merged_messages
        if request.use_rag:
            recent_user_msgs = [m["content"] for m in merged_messages if m["role"] == "user"][-3:]
            rag_query = " ".join(recent_user_msgs)
            rag_context = await augment_with_docstore(rag_query, top_k=3)
            web_results = await _web_search(rag_query, top_k=3) if request.web_search else []
            context_parts = []
            if rag_context:
                context_parts.append(f"Contexte documentaire:\n{rag_context}")
            if web_results:
                context_parts.append(f"Sources web:\n{_format_web_results(web_results)}")
            if context_parts:
                system_prompt = (
                    "Tu es un assistant qui répond aux questions en utilisant les sources ci-dessous. "
                    "Utilise ces informations pour répondre de manière précise et cite tes sources. "
                    "Si les sources ne contiennent pas la réponse, dis-le clairement.\n\n"
                    + "\n\n".join(context_parts)
                )
                augmented_messages = [
                    {"role": "system", "content": system_prompt},
                    *merged_messages,
                ]

        # Apply sliding window before sending to LLM
        augmented_messages = _trim_messages(augmented_messages)

        result = await chat_service.chat(
            messages=augmented_messages,
            model=request.model,
            provider=request.provider,
            use_rag=request.use_rag,
            strip_thinking=request.strip_thinking,
        )

        # Persist updated conversation to Redis
        if request.session_id:
            user_msg_content = request.messages[-1]["content"] if request.messages else ""
            assistant_content = result["content"]
            updated = merged_messages + [
                {"role": "user", "content": user_msg_content},
                {"role": "assistant", "content": assistant_content},
            ]
            # Avoid duplicate if user message was already in merged_messages
            last_non_system = [m for m in merged_messages if m["role"] != "system"]
            if last_non_system and last_non_system[-1].get("role") == "user" and last_non_system[-1].get("content") == user_msg_content:
                updated = merged_messages + [{"role": "assistant", "content": assistant_content}]
            await _save_session(request.session_id, updated)

        return ChatResponse(
            content=result["content"],
            model=result.get("model", request.model),
            provider=result.get("provider", request.provider or "auto"),
            usage=ChatResponse.Usage.model_validate(result.get("usage", {})),
            trace_id=result.get("trace_id", ""),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat endpoint."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    # Load session history before entering the generator (async context safe)
    history: list[dict] = []
    if request.session_id:
        history = await _load_session(request.session_id)

    # Merge history + current messages
    if history:
        last_history_content = history[-1].get("content", "") if history else ""
        first_request_content = request.messages[0].get("content", "") if request.messages else ""
        if last_history_content == first_request_content:
            merged_messages = history
        else:
            merged_messages = history + list(request.messages)
    else:
        merged_messages = list(request.messages)

    async def event_generator():
        try:
            # Context-aware RAG: use last 3 user messages as search query
            augmented_messages = merged_messages
            if request.use_rag:
                recent_user_msgs = [m["content"] for m in merged_messages if m["role"] == "user"][-3:]
                rag_query = " ".join(recent_user_msgs)
                rag_context = await augment_with_docstore(rag_query, top_k=3)
                web_results = await _web_search(rag_query, top_k=3) if request.web_search else []
                context_parts = []
                if rag_context:
                    context_parts.append(f"Contexte documentaire:\n{rag_context}")
                if web_results:
                    context_parts.append(f"Sources web:\n{_format_web_results(web_results)}")
                if context_parts:
                    system_prompt = (
                        "Tu es un assistant qui répond aux questions en utilisant les sources ci-dessous. "
                        "Utilise ces informations pour répondre de manière précise et cite tes sources. "
                        "Si les sources ne contiennent pas la réponse, dis-le clairement.\n\n"
                        + "\n\n".join(context_parts)
                    )
                    augmented_messages = [
                        {"role": "system", "content": system_prompt},
                        *merged_messages,
                    ]

            # Apply sliding window before sending to LLM
            augmented_messages = _trim_messages(augmented_messages)

            full_response = ""
            async for chunk in chat_service.stream_chat(
                messages=augmented_messages,
                model=request.model,
                provider=request.provider,
                strip_thinking=request.strip_thinking,
            ):
                delta = chunk.content or ""
                if delta:
                    full_response += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"

            # Persist updated conversation to Redis
            if request.session_id:
                user_msg_content = request.messages[-1]["content"] if request.messages else ""
                last_non_system = [m for m in merged_messages if m["role"] != "system"]
                if last_non_system and last_non_system[-1].get("role") == "user" and last_non_system[-1].get("content") == user_msg_content:
                    updated = merged_messages + [{"role": "assistant", "content": full_response}]
                else:
                    updated = merged_messages + [
                        {"role": "user", "content": user_msg_content},
                        {"role": "assistant", "content": full_response},
                    ]
                await _save_session(request.session_id, updated)

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    """Scraper une page via Camoufox."""
    if not browser_service:
        raise HTTPException(status_code=500, detail="Browser service not initialized")

    try:
        result = await browser_service.scrape(
            url=request.url,
            selector=request.selector,
            timeout_ms=request.timeout_ms,
        )
        return ScrapeResponse(**result)
    except BrowserDependencyMissingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except BrowserRemoteRunnerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except BrowserServiceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Scrape error: {e}")
        raise HTTPException(status_code=500, detail="scrape failed") from e


class FeedbackRequest(BaseModel):
    trace_id: str
    score: float = Field(ge=0, le=1)
    comment: str | None = None

@app.get("/alerts")
async def get_alerts(tail: int = 20):
    alerts_file = os.path.expanduser("~/.nc_rag_alerts.jsonl")
    if not os.path.exists(alerts_file):
        return {"alerts": [], "count": 0}
    with open(alerts_file) as f:
        lines = f.readlines()
    entries = []
    for line in lines[-tail:]:
        try:
            entries.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    return {"alerts": list(reversed(entries)), "count": len(entries)}


@app.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    from life_core.langfuse_tracing import score_trace
    score_trace(
        trace_id=req.trace_id,
        name="user-feedback",
        value=req.score,
        comment=req.comment,
    )
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# OpenAI-compat shim (V1.6 Phase D1)
# -----------------------------------------------------------------------------
# Bypasser containers (dolibarr, browser-use, meet, suite-*) point their
# OPENAI_API_BASE at http://life-core:8000/v1 and inherit routing, fallback,
# and Langfuse tracing instead of calling cloud providers directly.
import time as _time
import uuid as _uuid


class _OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class _OpenAIEmbeddingRequest(BaseModel):
    """OpenAI-compat /v1/embeddings request body.

    ``input`` matches OpenAI's shape: a single string or a list of
    strings. V1.8 does not yet accept token-id arrays (V1.9 backlog).
    ``model`` is a free-form id; current V1.8 routing ignores it and
    always calls the Tower TEI backend. Schema validation still
    requires the field for parity with OpenAI clients.
    """

    input: str | list[str]
    model: str | None = None
    user: str | None = None
    encoding_format: str | None = None  # "float" | "base64" — V1.9


async def stream_backend_chunks(payload: dict):
    """Stream OpenAI-compat chunks through the shared ChatService.

    Routes the streaming call via ``chat_service.chat_stream`` so the
    LiteLLM provider resolves the correct ``api_base``/``api_key`` per
    model (same path the non-stream shim uses). Yields UTF-8 SSE frames
    terminated by a ``data: [DONE]`` sentinel, suitable for a
    ``StreamingResponse(media_type="text/event-stream")``.
    """
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    messages = payload["messages"]
    model = payload.get("model") or DEFAULT_CHAT_MODEL

    forward_kwargs: dict = {}
    for key in ("tools", "tool_choice", "temperature", "max_tokens"):
        if payload.get(key) is not None:
            forward_kwargs[key] = payload[key]

    try:
        async for chunk in chat_service.chat_stream(
            messages=messages,
            model=model,
            **forward_kwargs,
        ):
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.exception("stream_backend_chunks failed: %s", exc)
        err = json.dumps({"error": str(exc)})
        yield f"data: {err}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def stream_backend_chat(payload: dict) -> StreamingResponse:
    """Return a StreamingResponse that relays backend SSE frames.

    Forces ``stream=True`` on the outbound payload and sets
    ``X-Accel-Buffering: no`` so Traefik / nginx do not buffer.
    """
    payload["stream"] = True
    return StreamingResponse(
        stream_backend_chunks(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def embed_backend(texts: list[str]) -> list[list[float]]:
    """V1.8 Wave B axis 10 — single embedding backend call.

    Delegates to ``life_core.rag.pipeline.EmbeddingModel.embed_batch``
    which already implements the TEI → sentence-transformers cascade
    gated on ``EMBED_URL``. Returns one float vector per input string
    in the same order as ``texts``.
    """
    from life_core.rag.pipeline import EmbeddingModel

    model = EmbeddingModel()
    return await model.embed_batch(texts)


async def call_backend_chat(payload: dict) -> dict:
    """Forward an OpenAI-compat chat payload to the router backend.

    Patched in tests to capture or override the forward call. In
    production, this wraps ``chat_service.chat()`` and repacks the
    response into the OpenAI-compat envelope. ``tools``,
    ``tool_choice``, ``temperature`` and ``max_tokens`` are forwarded
    as kwargs so they reach ``litellm.acompletion`` unchanged.
    """
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    messages = payload["messages"]
    model = payload.get("model") or DEFAULT_CHAT_MODEL

    forward_kwargs: dict = {}
    for key in ("tools", "tool_choice", "temperature", "max_tokens"):
        if payload.get(key) is not None:
            forward_kwargs[key] = payload[key]

    result = await chat_service.chat(
        messages=messages,
        model=model,
        provider=None,
        use_rag=False,
        strip_thinking=False,
        **forward_kwargs,
    )
    usage = result.get("usage", {})
    tool_calls = result.get("tool_calls")
    message: dict = {
        "role": "assistant",
        "content": result["content"],
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason = "tool_calls" if tool_calls else "stop"
    return {
        "id": f"chatcmpl-{_uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": result.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
        },
    }


@app.get("/v1/models", dependencies=V1_AUTH_DEPS)
async def openai_compat_models():
    """OpenAI-compat list shape backed by the same provider scan as
    /models. V1.7 Track II Task 13: each entry carries a
    ``capabilities`` list (chat/embedding/vision) derived from an
    explicit override table with heuristic fallback on the id."""
    from life_core.models.capabilities import guess_capabilities

    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")
    models: set[str] = set()
    for provider_id in router.list_available_providers():
        try:
            provider = router.providers[provider_id]
            provider_models = await provider.list_models()
            models.update(provider_models)
        except Exception as e:
            logger.warning("Failed to list models for %s: %s", provider_id, e)
    now = int(_time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "life-core",
                "capabilities": guess_capabilities(m),
            }
            for m in sorted(models)
        ],
    }


@app.post(
    "/v1/chat/completions",
    dependencies=V1_AUTH_DEPS,
)
async def openai_compat_chat(req: _OpenAIChatRequest):
    """OpenAI-compat non-streaming chat. Consumers that set
    OPENAI_API_BASE=http://life-core:8000/v1 get routing + fallback +
    Langfuse for free. Streaming kept on the existing /chat/stream
    path (SSE shape differs from OpenAI's chunked delta).

    When the client sends ``tools`` + ``tool_choice`` the payload is
    relayed verbatim to the backend router. Qwen3.6-35B-A3B speaks
    OpenAI function-calling natively via its chat template.
    """
    requested_model = resolve_model_alias(req.model or DEFAULT_CHAT_MODEL)
    payload: dict = {
        "model": requested_model,
        "messages": req.messages,
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.tools is not None:
        payload["tools"] = req.tools
    if req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice

    if req.stream:
        return await stream_backend_chat(payload)

    return await call_backend_chat(payload)


@app.post(
    "/v1/embeddings",
    dependencies=V1_AUTH_DEPS,
)
async def openai_compat_embeddings(req: _OpenAIEmbeddingRequest):
    """V1.8 Wave B axis 10 — OpenAI-compat /v1/embeddings.

    Accepts ``input`` as a string or list of strings and returns
    ``data[].embedding`` float arrays with ``usage.prompt_tokens``.
    Single backend (Tower TEI via EMBED_URL) in V1.8; multi-backend
    routing is V1.9 backlog.
    """
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = list(req.input)

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="'input' must be a non-empty string or list of strings",
        )

    try:
        vectors = await embed_backend(texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embeddings backend failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    if len(vectors) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=(
                f"embedding backend returned {len(vectors)} vectors "
                f"for {len(texts)} inputs"
            ),
        )

    # OpenAI counts prompt_tokens via tiktoken; V1.8 uses a cheap
    # whitespace-word proxy (V1.9 will swap in tiktoken for parity).
    prompt_tokens = sum(max(1, len(t.split())) for t in texts)

    return {
        "object": "list",
        "model": req.model or "tei/default",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vectors[i],
            }
            for i in range(len(vectors))
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


# -----------------------------------------------------------------------------
# V1.7 Track II — /traces Langfuse proxy with cursor pagination.
# Side-emits one langfuse.trace SSE event per newly-seen trace id.
# -----------------------------------------------------------------------------
@app.get("/traces", dependencies=V1_AUTH_DEPS)
async def traces_endpoint(
    cursor: str | None = None,
    limit: int = 50,
):
    """V1.7 Track II — Langfuse traces direct, cursor pagination."""
    from life_core.integrations.langfuse import fetch_traces

    return await fetch_traces(cursor=cursor, limit=limit)


# -----------------------------------------------------------------------------
# V1.7 Track II — /governance endpoint.
# Aggregates branch protection + open PR counts for F4L repos on
# both Forgejo (source of truth) and GitHub (legacy mirror). Cache 60s.
# -----------------------------------------------------------------------------
@app.get("/governance", dependencies=V1_AUTH_DEPS)
async def governance_endpoint():
    """V1.7 Track II — branch protection + open PRs, cached 60s."""
    from life_core.integrations.governance import get_governance

    return await get_governance()


# -----------------------------------------------------------------------------
# V1.7 Track II Task 10 — /schematic endpoint (Forgejo KiCad projects).
# Lists factory-4-life repos whose root contains a .kicad_pro file.
# Cached 60 s. Auth: LIFE_INTERNAL_BEARER or Keycloak JWT.
# -----------------------------------------------------------------------------
@app.get("/schematic", dependencies=V1_AUTH_DEPS)
async def schematic_endpoint():
    """V1.7 Track II — Forgejo KiCad projects list."""
    from life_core.integrations.forgejo_schematic import (
        list_kicad_projects,
    )

    return await list_kicad_projects()


# -----------------------------------------------------------------------------
# V1.7 Track II Task 9 — /workflow passthrough to engine.saillant.cc.
# Forwards GET/POST with caller Authorization header verbatim.
# -----------------------------------------------------------------------------
from fastapi import Response as _FastAPIResponse  # noqa: E402


@app.api_route(
    "/workflow/{subpath:path}",
    methods=["GET", "POST"],
)
async def workflow_endpoint(
    subpath: str,
    request: Request,
    _bearer: None = Depends(validate_life_internal_bearer),
):
    """V1.7 Track II — proxy to engine.saillant.cc."""
    from life_core.integrations.workflow_proxy import proxy

    status_code, body = await proxy(request, subpath)
    return _FastAPIResponse(
        content=json.dumps(body),
        media_type="application/json",
        status_code=status_code,
    )


# -----------------------------------------------------------------------------
# V1.7 Track II — unified SSE /events stream (replaces /health, /stats,
# /goose-stats polling).
# -----------------------------------------------------------------------------
@app.get("/events")
async def events_stream(
    request: Request,
    _bearer: None = Depends(validate_life_internal_bearer),
) -> EventSourceResponse:
    """Unified SSE stream. Clients filter by `event:` field locally.

    Auth: LIFE_INTERNAL_BEARER (Bucket A) OR Keycloak JWT (Bucket B)
    as in the rest of /v1 — the bearer dependency short-circuits on
    `X-Auth-Mode: bearer` so JWT can be validated upstream by
    `validate_keycloak_jwt` when wired.
    """
    broker = get_broker()
    queue = broker.subscribe()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keep-alive comment so intermediaries do not close.
                    yield {"event": "keepalive", "data": "{}"}
                    continue
                yield event.to_sse()
        finally:
            broker.unsubscribe(queue)

    return EventSourceResponse(gen())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "life_core.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
