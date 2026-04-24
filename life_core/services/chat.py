"""Services pour life-core."""

from __future__ import annotations

import logging
import time
from typing import Any

from life_core.cache import MultiTierCache
from life_core.rag import RAGPipeline
from life_core.router import Router
from life_core.telemetry import create_llm_instruments

logger = logging.getLogger("life_core.services")

_llm_metrics = None


def _get_llm_metrics() -> dict:
    global _llm_metrics
    if _llm_metrics is None:
        _llm_metrics = create_llm_instruments()
    return _llm_metrics


class ChatService:
    """Service de chat du coeur."""

    def __init__(
        self,
        router: Router,
        cache: MultiTierCache | None = None,
        rag: RAGPipeline | None = None,
        trace_emitter=None,
    ):
        """
        Créer le service de chat.

        Args:
            router: Routeur LLM
            cache: Cache optionnel
            rag: Pipeline RAG optionnel
            trace_emitter: TraceEmitter optionnel (V1.8 axis 5)
        """
        self.router = router
        self.cache = cache or MultiTierCache()
        self.rag = rag
        self.trace_emitter = trace_emitter
        self.stats = {"requests": 0, "cache_hits": 0}

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        provider: str | None = None,
        use_rag: bool = False,
        deliverable_slug: str = "adhoc",
        **kwargs
    ) -> dict[str, Any]:
        """
        Échanger un message avec un LLM.

        Args:
            messages: Liste de messages
            model: Modèle à utiliser
            provider: Provider spécifique (optionnel)
            use_rag: Utiliser le RAG (optionnel)
            deliverable_slug: slug pour inner_trace (V1.8 axis 5)
            **kwargs: Paramètres additionnels

        Returns:
            Dict avec 'content' (str) et 'usage' (dict)
        """
        self.stats["requests"] += 1
        import time as _time
        _start = _time.monotonic()

        # V1.8 axis 5 — record agent_run before the LLM call
        agent_run_id = None
        if self.trace_emitter is not None:
            agent_run_id = self.trace_emitter.record_agent_run(
                deliverable_slug=deliverable_slug,
                deliverable_type="chat",
                role="llm",
                outer_state_at_start="OPEN",
            )

        # Vérifier le cache — sauter pour les appels avec outils
        # (les tool_calls ne doivent pas être resservis depuis le cache)
        tools_present = bool(kwargs.get("tools"))
        cache_key = f"chat:{str(messages)[:100]}:{model}"
        if not tools_present:
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

        # Langfuse prompt versioning — inject system prompt if available
        from life_core.langfuse_tracing import get_langfuse_prompt
        prompt = get_langfuse_prompt("chat-system-prompt")
        if prompt:
            system_content = prompt.compile()
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": system_content}] + list(messages)

        # Appeler le routeur
        metrics = _get_llm_metrics()
        _llm_start = time.monotonic()
        try:
            response = await self.router.send(
                messages=messages,
                model=model,
                provider=provider,
                **kwargs
            )
        except Exception:
            metrics["llm_errors"].add(1, {"provider": provider or "unknown"})
            raise
        _llm_duration_ms = (time.monotonic() - _llm_start) * 1000
        _used_provider = getattr(response, "provider", provider or "unknown")
        _used_model = getattr(response, "model", model)
        metrics["llm_calls"].add(1, {"provider": _used_provider, "model": _used_model})
        metrics["llm_duration"].record(_llm_duration_ms, {"provider": _used_provider})

        # V1.8 axis 5 — record generation_run after a successful call
        if self.trace_emitter is not None and agent_run_id is not None:
            _usage = getattr(response, "usage", {}) or {}
            if not isinstance(_usage, dict):
                _usage = {
                    "prompt_tokens": getattr(_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(_usage, "completion_tokens", 0),
                }
            self.trace_emitter.record_generation_run(
                agent_run_id=agent_run_id,
                attempt_number=1,
                llm_model=getattr(response, "model", model),
                tokens_in=int(_usage.get("prompt_tokens", 0) or 0),
                tokens_out=int(_usage.get("completion_tokens", 0) or 0),
                cost_usd=float(getattr(response, "cost_usd", 0.0) or 0.0),
                status="success",
            )

        # Estimate cost from token usage
        usage = getattr(response, "usage", None) or {}
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

        _COST_PER_1M = {
            "vllm": {"input": 0.0, "output": 0.0},
            "ollama": {"input": 0.0, "output": 0.0},
            "anthropic": {"input": 3.0, "output": 15.0},
            "openai": {"input": 2.5, "output": 10.0},
        }
        _rates = _COST_PER_1M.get(_used_provider, {"input": 0.0, "output": 0.0})
        _cost = (prompt_tokens * _rates["input"] + completion_tokens * _rates["output"]) / 1_000_000
        if _cost > 0:
            metrics["llm_cost"].add(_cost, {"provider": _used_provider, "model": _used_model})

        # Extract OTEL trace_id for client correlation
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx.trace_id else ""

        result = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage if hasattr(response, "usage") else {},
            "trace_id": trace_id,
            "tool_calls": getattr(response, "tool_calls", None),
        }

        # Auto-scoring for Langfuse
        from life_core.langfuse_tracing import score_trace
        if trace_id:
            duration_s = _time.monotonic() - _start
            latency_score = max(0.0, 1.0 - (duration_s / 30.0))
            score_trace(trace_id=trace_id, name="latency", value=round(latency_score, 3))

        # Cacher la réponse — sauter pour les appels avec outils
        if not tools_present:
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

    async def chat_stream(
        self,
        *,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
    ):
        """Stream OpenAI-compat chunk dicts via the LiteLLM provider.

        Mirrors ``chat()`` so the streaming path inherits the same
        provider configuration (api_base, api_key, metadata) as the
        non-stream path. Each yielded value is a plain dict shaped like
        an OpenAI ``chat.completion.chunk`` so the caller can relay it
        verbatim to a downstream SSE consumer.
        """
        self.stats["requests"] += 1

        target_id = provider or self.router.primary_provider
        if not target_id or target_id not in self.router.providers:
            raise ValueError(
                f"No provider available for streaming (requested={provider})"
            )
        target = self.router.providers[target_id]
        stream_fn = getattr(target, "stream_openai_chunks", None)
        if stream_fn is None:
            raise ValueError(
                f"Provider {target_id} does not support OpenAI-compat streaming"
            )

        call_kwargs: dict[str, Any] = {}
        if tools is not None:
            call_kwargs["tools"] = tools
        if tool_choice is not None:
            call_kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        async for chunk in stream_fn(
            messages=messages, model=model, **call_kwargs
        ):
            yield chunk

    def get_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques du service."""
        return {
            **self.stats,
            "cache_stats": self.cache.get_stats(),
            "rag_stats": self.rag.get_stats() if self.rag else None,
        }
