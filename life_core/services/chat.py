"""Services pour life-core."""

from __future__ import annotations

import logging
from typing import Any

from life_core.cache import MultiTierCache
from life_core.rag import RAGPipeline
from life_core.router import Router

logger = logging.getLogger("life_core.services")


class ChatService:
    """Service de chat du coeur."""

    def __init__(
        self,
        router: Router,
        cache: MultiTierCache | None = None,
        rag: RAGPipeline | None = None,
    ):
        """
        Créer le service de chat.
        
        Args:
            router: Routeur LLM
            cache: Cache optionnel
            rag: Pipeline RAG optionnel
        """
        self.router = router
        self.cache = cache or MultiTierCache()
        self.rag = rag
        self.stats = {"requests": 0, "cache_hits": 0}

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        provider: str | None = None,
        use_rag: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        """
        Échanger un message avec un LLM.

        Args:
            messages: Liste de messages
            model: Modèle à utiliser
            provider: Provider spécifique (optionnel)
            use_rag: Utiliser le RAG (optionnel)
            **kwargs: Paramètres additionnels

        Returns:
            Dict avec 'content' (str) et 'usage' (dict)
        """
        self.stats["requests"] += 1

        # Vérifier le cache
        cache_key = f"chat:{str(messages)[:100]}:{model}"
        cached = await self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for message")
            return cached

        # Augmenter le context avec RAG si demandé
        if use_rag and self.rag:
            query = messages[-1].get("content", "")
            context = await self.rag.augment_context(query, top_k=3)
            if context:
                messages = [
                    *messages[:-1],
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ]

        # Appeler le routeur
        response = await self.router.send(
            messages=messages,
            model=model,
            provider=provider,
            **kwargs
        )

        # Extract OTEL trace_id for client correlation
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else ""

        result = {
            "content": response.content,
            "usage": response.usage if hasattr(response, "usage") else {},
            "trace_id": trace_id,
        }

        # Cacher la réponse
        await self.cache.set(cache_key, result, ttl=3600)

        return result

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        provider: str | None = None,
        **kwargs
    ):
        """Streamer une réponse de chat."""
        self.stats["requests"] += 1

        async for chunk in self.router.stream(
            messages=messages,
            model=model,
            provider=provider,
            **kwargs
        ):
            yield chunk

    def get_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques du service."""
        return {
            **self.stats,
            "cache_stats": self.cache.get_stats(),
            "rag_stats": self.rag.get_stats() if self.rag else None,
        }
