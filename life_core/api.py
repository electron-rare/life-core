"""API FastAPI pour life-core."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import json

import httpx
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
from life_core.models_api import models_router
from life_core.audit_api import audit_router
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

# Global instances
router: Router | None = None
cache: MultiTierCache | None = None
rag: RAGPipeline | None = None
chat_service: ChatService | None = None
browser_service: BrowserService | None = None


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
    ollama_model_aliases: set[str] = set()
    for env_var, default_models in DEFAULT_MODELS.items():
        if os.getenv(env_var):
            models += default_models

    ollama_api_base = os.getenv("OLLAMA_URL")
    ollama_models_env = os.getenv("OLLAMA_MODELS", "ollama/llama3")
    if ollama_api_base:
        ollama_models = [model.strip() for model in ollama_models_env.split(",") if model.strip()]
        models += ollama_models
        ollama_model_aliases = {model.removeprefix("ollama/") for model in ollama_models}

    ollama_remote = os.getenv("OLLAMA_REMOTE_URL")
    if ollama_remote and not ollama_api_base:
        ollama_api_base = ollama_remote
        ollama_models = [model.strip() for model in ollama_models_env.split(",") if model.strip()]
        models += ollama_models
        ollama_model_aliases = {model.removeprefix("ollama/") for model in ollama_models}

    # --- vLLM provider (KXKM-AI GPU) ---
    vllm_base = os.getenv("VLLM_BASE_URL")
    vllm_models_str = os.getenv("VLLM_MODELS", "")
    vllm_models: set[str] = set()
    if vllm_base and vllm_models_str:
        vllm_models = {m.strip() for m in vllm_models_str.split(",")}
        models += list(vllm_models)
        logger.info("vLLM models registered: %s via %s", vllm_models, vllm_base)

    if override := os.getenv("LITELLM_MODELS"):
        models = [m.strip() for m in override.split(",")]

    if models:
        provider = LiteLLMProvider(
            models=models,
            ollama_api_base=ollama_api_base,
            ollama_model_aliases=ollama_model_aliases,
            vllm_api_base=vllm_base,
            vllm_models=vllm_models,
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

    # Wire Redis to conversations
    if cache and hasattr(cache, '_redis') and cache._redis:
        set_redis(cache._redis)
    else:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis as redis_lib
                set_redis(redis_lib.from_url(redis_url))
            except Exception:
                pass

    providers = router.list_available_providers()
    logger.info(f"life-core initialized with {len(providers)} providers: {providers}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down life-core API")
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
    """Réponse de health check."""
    status: str
    providers: list[str]
    backends: list[str] = []
    cache_available: bool


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
    """Vérifier la santé de l'API."""
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")
    
    providers = router.list_available_providers()

    # Detect real backends behind LiteLLM proxy
    backends = []
    vllm_url = os.environ.get("VLLM_BASE_URL", "")
    ollama_url = os.environ.get("OLLAMA_URL", "")
    if vllm_url:
        backends.append("vllm")
    if ollama_url:
        backends.append("ollama")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(key, "")
        if val:
            backends.append(key.replace("_API_KEY", "").lower())

    return HealthResponse(
        status="ok",
        providers=providers,
        backends=backends,
        cache_available=cache is not None,
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
        # Augment with Cils docstore if RAG enabled
        rag_context = ""
        if request.use_rag:
            user_msg = request.messages[-1]["content"] if request.messages else ""
            rag_context = await augment_with_docstore(user_msg, top_k=3)
            web_results = await _web_search(user_msg, top_k=3) if request.web_search else []
            context_parts = []
            if rag_context:
                context_parts.append(f"Contexte documentaire:\n{rag_context}")
            if web_results:
                context_parts.append(f"Sources web:\n{_format_web_results(web_results)}")
            if context_parts:
                augmented_messages = [
                    {"role": "system", "content": "\n\n".join(context_parts)},
                    *request.messages,
                ]
            else:
                augmented_messages = request.messages
        else:
            augmented_messages = request.messages

        result = await chat_service.chat(
            messages=augmented_messages,
            model=request.model,
            provider=request.provider,
            use_rag=request.use_rag,
        )

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

    async def event_generator():
        try:
            augmented_messages = request.messages
            if request.use_rag:
                user_msg = request.messages[-1]["content"] if request.messages else ""
                rag_context = await augment_with_docstore(user_msg, top_k=3)
                web_results = await _web_search(user_msg, top_k=3) if request.web_search else []
                context_parts = []
                if rag_context:
                    context_parts.append(f"Contexte documentaire:\n{rag_context}")
                if web_results:
                    context_parts.append(f"Sources web:\n{_format_web_results(web_results)}")
                if context_parts:
                    augmented_messages = [
                        {"role": "system", "content": "\n\n".join(context_parts)},
                        *request.messages,
                    ]

            async for chunk in chat_service.stream_chat(
                messages=augmented_messages,
                model=request.model,
                provider=request.provider,
            ):
                delta = chunk.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "life_core.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
