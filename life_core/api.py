"""API FastAPI pour life-core."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from life_core.cache import MultiTierCache
from life_core.rag import RAGPipeline
from life_core.rag.api import rag_router, set_rag_pipeline
from life_core.infra_api import infra_router
from life_core.traces_api import traces_router
from life_core.stats_api import stats_router, record_call
from life_core.logs_api import logs_router
from life_core.conversations_api import conversations_router, set_redis
from life_core.models_api import models_router
from life_core.router import LiteLLMProvider, Router
from life_core.services import ChatService
from life_core.langfuse_tracing import flush_langfuse, init_langfuse
from life_core.telemetry import init_telemetry
from life_core.docstore_client import augment_with_docstore

logger = logging.getLogger("life_core.api")

# Global instances
router: Router | None = None
cache: MultiTierCache | None = None
rag: RAGPipeline | None = None
chat_service: ChatService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'app."""
    global router, cache, rag, chat_service
    
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

    ollama_api_base = os.getenv("OLLAMA_URL")
    if ollama_api_base:
        ollama_models = os.getenv("OLLAMA_MODELS", "ollama/llama3").split(",")
        models += [m.strip() for m in ollama_models]

    ollama_remote = os.getenv("OLLAMA_REMOTE_URL")
    if ollama_remote and not ollama_api_base:
        ollama_api_base = ollama_remote
        ollama_models = os.getenv("OLLAMA_MODELS", "ollama/llama3").split(",")
        models += [m.strip() for m in ollama_models]

    if override := os.getenv("LITELLM_MODELS"):
        models = [m.strip() for m in override.split(",")]

    if models:
        provider = LiteLLMProvider(models=models, ollama_api_base=ollama_api_base)
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
app.include_router(traces_router)
app.include_router(stats_router)
app.include_router(logs_router)
app.include_router(conversations_router)
app.include_router(models_router)

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
class ChatRequest(BaseModel):
    """Requête de chat."""
    messages: list[dict[str, str]]
    model: str = "claude-3-5-sonnet-20241022"
    provider: str | None = None
    use_rag: bool = False


class ChatResponse(BaseModel):
    """Réponse de chat."""
    content: str
    model: str
    provider: str
    usage: dict[str, int] = {}
    trace_id: str = ""


class HealthResponse(BaseModel):
    """Réponse de health check."""
    status: str
    providers: list[str]
    cache_available: bool


class ModelsResponse(BaseModel):
    """Réponse des modèles disponibles."""
    models: list[str]


# Routes
@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifier la santé de l'API."""
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")
    
    providers = router.list_available_providers()

    return HealthResponse(
        status="ok",
        providers=providers,
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
            if rag_context:
                # Prepend context as system message
                augmented_messages = [
                    {"role": "system", "content": f"Contexte documentaire:\n{rag_context}"},
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
            model=request.model,
            provider=request.provider or "auto",
            trace_id=result.get("trace_id", ""),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "life_core.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
