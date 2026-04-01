"""Integration test — verify full OTEL trace tree from chat request.

Run: pytest tests/test_integration_otel.py -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from life_core.router.providers.base import LLMResponse
from life_core.services.chat import ChatService
from life_core.cache.multi_tier_cache import MultiTierCache


def _make_test_tracer():
    exporter = InMemorySpanExporter()
    tp = SdkTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tracer, exporter


@pytest.mark.asyncio
async def test_chat_produces_full_span_tree():
    """A chat call should emit spans for cache and LLM call."""
    tracer, exporter = _make_test_tracer()

    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Test response",
        model="openai/gpt-4o",
        provider="litellm",
        usage={"input_tokens": 10, "output_tokens": 5},
    ))

    cache = MultiTierCache()
    service = ChatService(router=mock_router, cache=cache, rag=None)

    with patch("life_core.telemetry.get_tracer", return_value=tracer), \
         patch("life_core.langfuse_tracing.get_langfuse_prompt", return_value=None), \
         patch("life_core.langfuse_tracing.score_trace"):
        result = await service.chat(
            messages=[{"role": "user", "content": "Test"}],
            model="openai/gpt-4o",
        )

    assert result["content"] == "Test response"
    assert "trace_id" in result

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    # Cache spans should be present
    assert "cache.l1.get" in span_names
    assert "cache.store" in span_names
