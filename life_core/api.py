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
from life_core.router import ClaudeProvider, GoogleProvider, GroqProvider, MistralProvider, OpenAIProvider, Router
from life_core.router.providers.ollama import OllamaProvider
from life_core.services import ChatService
from life_core.telemetry import init_telemetry

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

    # Initialiser le routeur
    router = Router()
    
    # Enregistrer les providers
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = ClaudeProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
        router.register_provider(claude, is_primary=True)
    
    if os.getenv("OPENAI_API_KEY"):
        openai = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
        router.register_provider(openai)
    
    if os.getenv("GOOGLE_API_KEY"):
        google = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
        router.register_provider(google)

    if os.getenv("MISTRAL_API_KEY"):
        mistral = MistralProvider(api_key=os.getenv("MISTRAL_API_KEY"))
        router.register_provider(mistral)

    if os.getenv("GROQ_API_KEY"):
        groq = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
        router.register_provider(groq)

    # Ollama local (Tower)
    ollama_url = os.environ.get("OLLAMA_URL")
    if ollama_url:
        router.register_provider(OllamaProvider(base_url=ollama_url, name="ollama"))
        logger.info(f"Registered Ollama provider at {ollama_url}")

    # Ollama remote (KXKM-AI via Tailscale)
    ollama_remote_url = os.environ.get("OLLAMA_REMOTE_URL")
    if ollama_remote_url:
        router.register_provider(OllamaProvider(base_url=ollama_remote_url, name="ollama-gpu"))
        logger.info(f"Registered Ollama GPU provider at {ollama_remote_url}")

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
    
    # Créer le service de chat
    chat_service = ChatService(router=router, cache=cache, rag=rag)
    
    providers = router.list_available_providers()
    logger.info(f"life-core initialized with {len(providers)} providers: {providers}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down life-core API")


# Créer l'app
app = FastAPI(
    title="life-core API",
    description="Life-core LLM router and RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        response_content = await chat_service.chat(
            messages=request.messages,
            model=request.model,
            provider=request.provider,
            use_rag=request.use_rag,
        )
        
        return ChatResponse(
            content=response_content,
            model=request.model,
            provider=request.provider or "auto",
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
