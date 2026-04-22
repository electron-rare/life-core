"""LiteLLM unified provider — single entry point for all LLM backends."""
import logging
from collections.abc import AsyncIterator

import litellm

from .base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """Routes all LLM calls through litellm.acompletion().

    Model names use LiteLLM format: "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", etc.
    API keys are read from standard env vars by LiteLLM (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
    """

    def __init__(
        self,
        models: list[str],
        ollama_api_base: str | None = None,
        ollama_model_aliases: set[str] | None = None,
        vllm_api_base: str | None = None,
        vllm_models: set[str] | None = None,
        local_llm_api_base: str | None = None,
        local_llm_models: set[str] | None = None,
        kiki_full_base_url: str | None = None,
        kiki_full_models: set[str] | None = None,
        **kwargs,
    ):
        super().__init__(provider_id="litellm", **kwargs)
        self.models = models
        self.ollama_api_base = ollama_api_base
        self.ollama_model_aliases = {
            model.removeprefix("ollama/") for model in (ollama_model_aliases or set())
        }
        self.vllm_api_base = vllm_api_base
        self.vllm_models = vllm_models or set()
        self.local_llm_api_base = local_llm_api_base
        self.local_llm_models = local_llm_models or set()
        self.kiki_full_base_url = kiki_full_base_url
        self.kiki_full_models = kiki_full_models or set()
        if self.kiki_full_models and not self.kiki_full_base_url:
            raise ValueError("kiki-* models listed but kiki_full_base_url not set")

    async def send(self, messages: list[dict], model: str, **kwargs) -> LLMResponse:
        resolved_model = self._resolve_model_name(model)
        call_kwargs = self._build_call_kwargs(resolved_model, kwargs)
        response = await litellm.acompletion(
            model=resolved_model,
            messages=messages,
            **call_kwargs,
        )
        return self._to_llm_response(response, model)

    async def stream(
        self, messages: list[dict], model: str, **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        resolved_model = self._resolve_model_name(model)
        call_kwargs = self._build_call_kwargs(resolved_model, kwargs)
        response = await litellm.acompletion(
            model=resolved_model,
            messages=messages,
            stream=True,
            **call_kwargs,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield LLMStreamChunk(
                    content=delta.content,
                    model=model,
                    finish_reason=chunk.choices[0].finish_reason,
                )

    async def health_check(self) -> bool:
        if not self.models:
            return False
        try:
            model = self._resolve_model_name(self.models[0])
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                **self._build_call_kwargs(model, {}),
            )
            return True
        except Exception as e:
            logger.warning("LiteLLM health check failed: %s", e)
            return False

    async def list_models(self) -> list[str]:
        return list(self.models)

    def _resolve_model_name(self, model: str) -> str:
        if model.startswith("ollama/"):
            return model
        if model in self.ollama_model_aliases:
            return f"ollama/{model}"
        if model in self.kiki_full_models:
            # kiki-router speaks OpenAI-compat; prefix routes via openai provider
            return f"openai/{model}"
        return model

    def _build_call_kwargs(self, model: str, extra: dict) -> dict:
        kwargs = dict(extra)
        if model.startswith("ollama/") and self.ollama_api_base:
            kwargs["api_base"] = self.ollama_api_base
        elif model.startswith("openai/kiki-") and self.kiki_full_base_url:
            # Route kiki-* to kiki-router (overrides global OPENAI_API_BASE)
            kwargs["api_base"] = self.kiki_full_base_url
        elif model in self.vllm_models and self.vllm_api_base:
            kwargs["api_base"] = self.vllm_api_base
        elif model in self.local_llm_models and self.local_llm_api_base:
            kwargs["api_base"] = self.local_llm_api_base

        # Inject OTEL trace context into LiteLLM metadata for Langfuse correlation
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.trace_id:
            metadata = kwargs.get("metadata", {})
            metadata["trace_id"] = format(ctx.trace_id, "032x")
            metadata["span_id"] = format(ctx.span_id, "016x")
            kwargs["metadata"] = metadata

        return kwargs

    def _to_llm_response(self, response, model: str) -> LLMResponse:
        choice = response.choices[0]
        usage = response.usage
        llm_response = LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            provider="litellm",
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )
        try:
            cost = litellm.completion_cost(completion_response=response)
            from life_core.telemetry import get_meter
            meter = get_meter()
            cost_counter = meter.create_counter("llm.cost", description="LLM call cost in USD")
            cost_counter.add(cost, {"model": response.model or model, "provider": "litellm"})
        except Exception:
            pass  # Cost calculation not critical
        return llm_response
