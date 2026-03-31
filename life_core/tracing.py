"""LLM call tracing — wraps provider calls with OpenTelemetry spans."""

from __future__ import annotations

import time
import logging
from typing import Any

from .telemetry import get_tracer, get_meter

logger = logging.getLogger("life_core.tracing")

# Metrics counters (lazy init)
_llm_calls = None
_llm_errors = None
_llm_duration = None


def _ensure_metrics():
    global _llm_calls, _llm_errors, _llm_duration
    if _llm_calls is None:
        meter = get_meter()
        _llm_calls = meter.create_counter("llm.calls", description="Total LLM provider calls")
        _llm_errors = meter.create_counter("llm.errors", description="Total LLM provider errors")
        _llm_duration = meter.create_histogram("llm.duration_ms", description="LLM call duration in ms")


async def traced_llm_call(
    provider_name: str,
    model: str,
    messages: list[dict[str, str]],
    call_fn,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute an LLM call with OpenTelemetry tracing and metrics."""
    _ensure_metrics()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "llm.call",
        attributes={
            "llm.provider": provider_name,
            "llm.model": model,
            "llm.message_count": len(messages),
        },
    ) as span:
        start = time.monotonic()
        try:
            result = await call_fn(messages=messages, model=model, **kwargs)
            duration_ms = (time.monotonic() - start) * 1000

            # Record metrics
            _llm_calls.add(1, {"provider": provider_name, "model": model})
            _llm_duration.record(duration_ms, {"provider": provider_name, "model": model})

            # Enrich span
            usage = result.get("usage", {})
            span.set_attribute("llm.duration_ms", duration_ms)
            span.set_attribute("llm.tokens.prompt", usage.get("prompt_tokens", 0))
            span.set_attribute("llm.tokens.completion", usage.get("completion_tokens", 0))
            span.set_attribute("llm.response_length", len(result.get("content", "")))

            return result

        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            _llm_errors.add(1, {"provider": provider_name, "model": model, "error": type(e).__name__})
            span.set_attribute("llm.error", str(e))
            span.set_status_code(2)  # ERROR
            raise
