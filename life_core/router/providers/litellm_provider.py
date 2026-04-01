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
        **kwargs,
    ):
        super().__init__(provider_id="litellm", **kwargs)
        self.models = models
        self.ollama_api_base = ollama_api_base

    async def send(self, messages: list[dict], model: str, **kwargs) -> LLMResponse:
        call_kwargs = self._build_call_kwargs(model, kwargs)
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            **call_kwargs,
        )
        return self._to_llm_response(response, model)

    async def stream(
        self, messages: list[dict], model: str, **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        call_kwargs = self._build_call_kwargs(model, kwargs)
        response = await litellm.acompletion(
            model=model,
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
            await litellm.acompletion(
                model=self.models[0],
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                **self._build_call_kwargs(self.models[0], {}),
            )
            return True
        except Exception as e:
            logger.warning("LiteLLM health check failed: %s", e)
            return False

    async def list_models(self) -> list[str]:
        return list(self.models)

    def _build_call_kwargs(self, model: str, extra: dict) -> dict:
        kwargs = dict(extra)
        if model.startswith("ollama/") and self.ollama_api_base:
            kwargs["api_base"] = self.ollama_api_base

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
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            provider="litellm",
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )
