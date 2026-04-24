"""LiteLLM unified provider — single entry point for all LLM backends."""
import logging
import os
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

    async def stream_openai_chunks(
        self, messages: list[dict], model: str, **kwargs
    ) -> AsyncIterator[dict]:
        """Stream raw OpenAI-compat chunk dicts from the backend.

        Unlike ``stream()``, this preserves the full chunk shape
        (tool_call deltas, finish_reason, index) so the caller can
        relay it verbatim to an OpenAI-compat client. Used by the
        ``/v1/chat/completions?stream=true`` shim path.
        """
        resolved_model = self._resolve_model_name(model)
        call_kwargs = self._build_call_kwargs(resolved_model, kwargs)
        response = await litellm.acompletion(
            model=resolved_model,
            messages=messages,
            stream=True,
            **call_kwargs,
        )
        async for chunk in response:
            yield self._chunk_to_openai_dict(chunk, model)

    @staticmethod
    def _chunk_to_openai_dict(chunk, model: str) -> dict:
        """Normalise a litellm stream chunk into an OpenAI-compat dict.

        Prefers pydantic ``model_dump`` when available (real litellm
        ``ModelResponseStream``), falls back to manual extraction for
        test doubles and exotic shapes.
        """
        dump = getattr(chunk, "model_dump", None)
        if callable(dump):
            try:
                data = dump(exclude_none=True)
                if isinstance(data, dict) and data.get("choices"):
                    # Non-streaming providers (e.g. kiki-router for
                    # kiki-meta-*) wrap the response as a single chunk
                    # with an empty delta but content inside ``message``.
                    # Merge the message back into delta so downstream
                    # shims emit real tokens.
                    for ch in data.get("choices", []) or []:
                        delta = ch.get("delta") or {}
                        message = ch.get("message") or {}
                        if not delta.get("content") and message.get("content"):
                            delta["content"] = message["content"]
                        if "role" not in delta and message.get("role"):
                            delta["role"] = message["role"]
                        if "tool_calls" not in delta and message.get("tool_calls"):
                            delta["tool_calls"] = message["tool_calls"]
                        ch["delta"] = delta
                    return data
            except Exception:  # noqa: BLE001
                pass
        # Manual fallback
        choices_out: list[dict] = []
        for idx, ch in enumerate(getattr(chunk, "choices", []) or []):
            delta = getattr(ch, "delta", None)
            delta_out: dict = {}
            if delta is not None:
                content = getattr(delta, "content", None)
                if content is not None:
                    delta_out["content"] = content
                role = getattr(delta, "role", None)
                if role is not None:
                    delta_out["role"] = role
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    delta_out["tool_calls"] = tool_calls
            # Fallback: non-streaming provider wrapped as a single chunk.
            # LiteLLM may emit chunks where delta is empty but message
            # carries the content (seen with kiki-router / kiki-meta-*
            # which return a non-stream response forwarded as one frame).
            if not delta_out.get("content"):
                message = getattr(ch, "message", None)
                if message is not None:
                    msg_content = getattr(message, "content", None)
                    if msg_content:
                        delta_out["content"] = msg_content
                    if "role" not in delta_out:
                        msg_role = getattr(message, "role", None)
                        if msg_role is not None:
                            delta_out["role"] = msg_role
                    if "tool_calls" not in delta_out:
                        msg_tool_calls = getattr(message, "tool_calls", None)
                        if msg_tool_calls:
                            delta_out["tool_calls"] = msg_tool_calls
            choices_out.append(
                {
                    "index": getattr(ch, "index", idx),
                    "delta": delta_out,
                    "finish_reason": getattr(ch, "finish_reason", None),
                }
            )
        return {
            "id": getattr(chunk, "id", ""),
            "object": getattr(chunk, "object", "chat.completion.chunk"),
            "created": getattr(chunk, "created", 0),
            "model": getattr(chunk, "model", None) or model,
            "choices": choices_out,
        }

    async def health_check(self) -> bool:
        if not self.models:
            return False
        # Prefer a local pool model (no cloud auth needed) to avoid a
        # false-positive "down" when cloud API keys are placeholders
        # (e.g. in dev/prod-local deployments).
        probe = self._pick_health_probe_model()
        try:
            resolved = self._resolve_model_name(probe)
            await litellm.acompletion(
                model=resolved,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                **self._build_call_kwargs(resolved, {}),
            )
            return True
        except Exception as e:
            logger.warning("LiteLLM health check failed: %s", e)
            return False

    def _pick_health_probe_model(self) -> str:
        """Return a model name preferring local pools over cloud.

        Cloud models fail when API keys are placeholders — check a model
        we can actually reach locally first. Priority: vLLM, local_llm,
        kiki-router, Ollama, then fallback on the first declared model.
        """
        for model in self.models:
            if model in self.vllm_models:
                return model
        for model in self.models:
            if model in self.local_llm_models:
                return model
        for model in self.models:
            if model in self.kiki_full_models:
                return model
        for model in self.models:
            if model.startswith("ollama/") or model in self.ollama_model_aliases:
                return model
        return self.models[0]

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
            # Route kiki-* to kiki-router (overrides global OPENAI_API_BASE).
            # Isolate api_key so a real sk-proj OpenAI key never leaks into
            # the local pool (the router accepts any bearer token).
            kwargs["api_base"] = self.kiki_full_base_url
            kwargs["api_key"] = os.environ.get("KIKI_FULL_API_KEY", "vllm-er-2026")
        elif model in self.vllm_models and self.vllm_api_base:
            # Override api_key for vLLM calls. Prevents the real OPENAI_API_KEY
            # (sk-proj-...) from being sent to the local llama-server, which
            # would 401 and mark the pool degraded.
            kwargs["api_base"] = self.vllm_api_base
            kwargs["api_key"] = os.environ.get("VLLM_API_KEY", "vllm-er-2026")
        elif model in self.local_llm_models and self.local_llm_api_base:
            kwargs["api_base"] = self.local_llm_api_base
            kwargs["api_key"] = os.environ.get("LOCAL_LLM_API_KEY", "vllm-er-2026")

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

    @staticmethod
    def _extract_tool_calls(choice) -> list | None:
        """Normalise LiteLLM tool_calls into OpenAI-compat dicts.

        Returns ``None`` when no tool calls are present so the field
        is omitted from the shim response body.
        """
        raw = getattr(choice.message, "tool_calls", None)
        if not raw:
            return None
        normalised: list[dict] = []
        for tc in raw:
            if isinstance(tc, dict):
                normalised.append(tc)
                continue
            fn = getattr(tc, "function", None)
            normalised.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", None),
                    },
                }
            )
        return normalised or None

    def _to_llm_response(self, response, model: str) -> LLMResponse:
        choice = response.choices[0]
        usage = response.usage
        tool_calls = self._extract_tool_calls(choice)
        llm_response = LLMResponse(
            content=choice.message.content or "",
            model=response.model or model,
            provider="litellm",
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            tool_calls=tool_calls,
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
