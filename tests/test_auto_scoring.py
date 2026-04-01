"""Tests for automatic Langfuse scoring in ChatService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router.providers.base import LLMResponse
from life_core.services.chat import ChatService

_FAKE_TRACE_ID = "a" * 32


def _make_mock_otel_span(trace_id_int: int):
    """Return a mock OTEL span whose context carries the given trace_id."""
    mock_ctx = MagicMock()
    mock_ctx.trace_id = trace_id_int
    mock_span = MagicMock()
    mock_span.get_span_context.return_value = mock_ctx
    return mock_span


@pytest.mark.asyncio
async def test_chat_sends_latency_score():
    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Response", model="openai/gpt-4o", provider="litellm",
        usage={"input_tokens": 10, "output_tokens": 5},
    ))

    service = ChatService(router=mock_router, cache=None, rag=None)

    # Provide a non-zero OTEL trace_id so the scoring guard fires.
    fake_trace_id_int = int(_FAKE_TRACE_ID, 16)
    mock_span = _make_mock_otel_span(fake_trace_id_int)

    with patch("life_core.langfuse_tracing.get_langfuse_prompt", return_value=None), \
         patch("life_core.langfuse_tracing.score_trace") as mock_score, \
         patch("opentelemetry.trace.get_current_span", return_value=mock_span):
        result = await service.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="openai/gpt-4o",
        )

    latency_calls = [c for c in mock_score.call_args_list if c[1].get("name") == "latency"]
    assert len(latency_calls) == 1
    assert 0 <= latency_calls[0][1]["value"] <= 1
