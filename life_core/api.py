"""API FastAPI pour life-core."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import json

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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

app.include_router(rag_router)
app.include_router(infra_router)
app.include_router(monitoring_router)
app.include_router(ws_alerts_router)
app.include_router(traces_router)
app.include_router(stats_router)
app.include_router(logs_router)
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(audit_router)
app.include_router(goose_router)
app.include_router(projects_router)
app.include_router(team_router)
app.include_router(config_router)

from life_core.routes.agents import router as agents_router
app.include_router(agents_router)

from life_core.providers_api import providers_router
app.include_router(providers_router)

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


class ChatRequest(BaseModel):
    """Requête de chat."""
    messages: list[dict[str, str]]
    model: str = "openai/qwen-14b-awq"
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


class HealthResponse(BaseModel):
    """Réponse de health check agrégée."""
    status: str  # "ok" | "degraded"
    providers: list[str]
    backends: list[str] = []
    cache_available: bool
    router_status: dict[str, bool] = {}
    issues: list[str] = []


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
@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifier la santé de l'API (agrégat runtime)."""
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")

    providers = router.list_available_providers()

    # Router runtime status (chaque provider répond ou non)
    try:
        router_status = router.get_provider_status()
    except Exception as exc:
        logger.warning("router.get_provider_status failed: %s", exc)
        router_status = {p: False for p in providers}

    # Detect real backends behind LiteLLM proxy
    backends = []
    vllm_url = os.environ.get("VLLM_BASE_URL", "")
    if vllm_url:
        backends.append("vllm")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY",
                "GROQ_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(key, ""):
            backends.append(key.replace("_API_KEY", "").lower())

    # Aggregate status
    issues: list[str] = []
    for name, ok in router_status.items():
        if not ok:
            issues.append(f"router:{name}:down")

    # Optional lightweight vLLM ping (timeout configurable via env)
    vllm_timeout_ms = int(os.environ.get("HEALTH_VLLM_TIMEOUT_MS", "500"))
    vllm_timeout_s = vllm_timeout_ms / 1000.0
    if vllm_url:
        try:
            async with httpx.AsyncClient(timeout=vllm_timeout_s) as client:
                r = await client.get(f"{vllm_url}/health")
                if r.status_code != 200:
                    logger.warning("vLLM health ping returned %s", r.status_code)
                    issues.append("backend:vllm:down")
        except Exception as exc:
            logger.warning("vLLM health ping failed: %s", exc)
            issues.append("backend:vllm:down")

    status = "ok" if not issues else "degraded"

    return HealthResponse(
        status=status,
        providers=providers,
        backends=backends,
        cache_available=cache is not None,
        router_status=router_status,
        issues=issues,
    )


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


@app.get("/stats")
async def stats():
    """Obtenir les statistiques."""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")
    
    try:
        chat_stats = chat_service.get_stats()
    except Exception:
        chat_stats = {}
    try:
        router_status = router.get_provider_status() if router else {}
    except Exception:
        router_status = {}

    return {
        "chat_service": chat_stats,
        "router": {"status": router_status},
    }


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
from typing import Literal
import time as _time
import uuid as _uuid


class _OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class _OpenAIChatRequest(BaseModel):
    model: str
    messages: list[_OpenAIMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


@app.get("/v1/models")
async def openai_compat_models():
    """OpenAI-compat list shape backed by the same provider scan as
    /models."""
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
            {"id": m, "object": "model", "created": now, "owned_by": "life-core"}
            for m in sorted(models)
        ],
    }


@app.post("/v1/chat/completions")
async def openai_compat_chat(req: _OpenAIChatRequest):
    """OpenAI-compat non-streaming chat. Consumers that set
    OPENAI_API_BASE=http://life-core:8000/v1 get routing + fallback +
    Langfuse for free. Streaming kept on the existing /chat/stream
    path (SSE shape differs from OpenAI's chunked delta)."""
    if req.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming on the compat shim is not implemented; use /chat/stream.",
        )
    if not chat_service:
        raise HTTPException(status_code=500, detail="Chat service not initialized")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = await chat_service.chat(
        messages=messages,
        model=req.model,
        provider=None,
        use_rag=False,
        strip_thinking=False,
    )
    usage = result.get("usage", {})
    return {
        "id": f"chatcmpl-{_uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": result.get("model", req.model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"],
                },
                "finish_reason": "stop",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "life_core.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
