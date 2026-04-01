"""Test that chat responses include OTEL trace_id."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace as otel_trace

from life_core.router.providers.base import LLMResponse
from life_core.services.chat import ChatService


@pytest.mark.asyncio
async def test_chat_returns_trace_id():
    """chat() should return trace_id from current OTEL span."""
    exporter = InMemorySpanExporter()
    tp = SdkTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tp)
    tracer = tp.get_tracer("test")

    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Hi", model="openai/gpt-4o", provider="litellm",
        usage={"input_tokens": 5, "output_tokens": 3},
    ))

    service = ChatService(router=mock_router, cache=None, rag=None)

    with tracer.start_as_current_span("test-chat"):
        result = await service.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="openai/gpt-4o",
        )

    assert "trace_id" in result
    assert len(result["trace_id"]) == 32

    otel_trace.set_tracer_provider(otel_trace.NoOpTracerProvider())


@pytest.mark.asyncio
async def test_chat_returns_empty_trace_id_without_otel():
    """chat() should return empty trace_id when no OTEL span is active."""
    otel_trace.set_tracer_provider(otel_trace.NoOpTracerProvider())

    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Hi", model="openai/gpt-4o", provider="litellm",
        usage={"input_tokens": 5, "output_tokens": 3},
    ))

    service = ChatService(router=mock_router, cache=None, rag=None)
    result = await service.chat(
        messages=[{"role": "user", "content": "Hi"}],
        model="openai/gpt-4o",
    )

    assert "trace_id" in result
