"""Tests for LiteLLMProvider — async native via litellm.acompletion()."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace as otel_trace

from life_core.router.providers.litellm_provider import LiteLLMProvider
from life_core.router.providers.base import LLMResponse, LLMStreamChunk


def _make_response(content="hello", model="openai/gpt-4o", prompt_tokens=10, completion_tokens=5):
    """Build a minimal mock litellm response object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


def _make_stream_chunk(content, model="openai/gpt-4o", finish_reason=None):
    delta = MagicMock()
    delta.content = content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


# ---------------------------------------------------------------------------
# 1. send() returns proper LLMResponse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_returns_llm_response():
    provider = LiteLLMProvider(models=["openai/gpt-4o"])
    mock_response = _make_response(content="bonjour", model="openai/gpt-4o")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        result = await provider.send(
            messages=[{"role": "user", "content": "hello"}],
            model="openai/gpt-4o",
        )

    assert isinstance(result, LLMResponse)
    assert result.content == "bonjour"
    assert result.model == "openai/gpt-4o"
    assert result.provider == "litellm"
    assert result.usage["input_tokens"] == 10
    assert result.usage["output_tokens"] == 5


# ---------------------------------------------------------------------------
# 2. send() with ollama model passes api_base
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_with_ollama_passes_api_base():
    provider = LiteLLMProvider(
        models=["ollama/llama3"],
        ollama_api_base="http://localhost:11434",
    )
    mock_response = _make_response(content="pong", model="ollama/llama3")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        await provider.send(
            messages=[{"role": "user", "content": "ping"}],
            model="ollama/llama3",
        )

    _, kwargs = mock_call.call_args
    assert kwargs.get("api_base") == "http://localhost:11434"


# ---------------------------------------------------------------------------
# 3. send() propagates exceptions from litellm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_error_raises():
    provider = LiteLLMProvider(models=["openai/gpt-4o"])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = RuntimeError("rate limit")
        with pytest.raises(RuntimeError, match="rate limit"):
            await provider.send(
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
            )


# ---------------------------------------------------------------------------
# 4. stream() yields LLMStreamChunk objects
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_yields_chunks():
    provider = LiteLLMProvider(models=["openai/gpt-4o"])

    chunks = [
        _make_stream_chunk("hel"),
        _make_stream_chunk("lo"),
        _make_stream_chunk("", finish_reason="stop"),
    ]

    async def _async_iter(_chunks):
        for c in _chunks:
            yield c

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _async_iter(chunks)
        result_chunks = []
        async for chunk in provider.stream(
            messages=[{"role": "user", "content": "hello"}],
            model="openai/gpt-4o",
        ):
            result_chunks.append(chunk)

    # Only chunks with non-empty content (or finish_reason) should be yielded
    assert len(result_chunks) >= 2
    assert all(isinstance(c, LLMStreamChunk) for c in result_chunks)
    assert result_chunks[0].content == "hel"
    assert result_chunks[1].content == "lo"
    assert all(c.model == "openai/gpt-4o" for c in result_chunks)


# ---------------------------------------------------------------------------
# 5. health_check() returns True on success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_success():
    provider = LiteLLMProvider(models=["openai/gpt-4o"])
    mock_response = _make_response(content="ok", model="openai/gpt-4o")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_response
        result = await provider.health_check()

    assert result is True


# ---------------------------------------------------------------------------
# 6. health_check() returns False on failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_failure():
    provider = LiteLLMProvider(models=["openai/gpt-4o"])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("connection refused")
        result = await provider.health_check()

    assert result is False


# ---------------------------------------------------------------------------
# 7. health_check() returns False with no models
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_no_models():
    provider = LiteLLMProvider(models=[])
    result = await provider.health_check()
    assert result is False


# ---------------------------------------------------------------------------
# 8. list_models() returns configured model list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models():
    models = ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", "ollama/llama3"]
    provider = LiteLLMProvider(models=models)
    result = await provider.list_models()
    assert result == models


# ---------------------------------------------------------------------------
# 9. send() injects OTEL trace_id and span_id into metadata for Langfuse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_passes_otel_trace_id_in_metadata():
    """send() should inject current OTEL trace_id into litellm metadata."""
    exporter = InMemorySpanExporter()
    tp = SdkTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tp)
    tracer = tp.get_tracer("test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
    mock_response.model = "openai/gpt-4o"
    mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    provider = LiteLLMProvider(models=["openai/gpt-4o"])

    with tracer.start_as_current_span("test-span"):
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await provider.send(
                messages=[{"role": "user", "content": "Hi"}],
                model="openai/gpt-4o",
            )

    _, kwargs = mock_call.call_args
    assert "metadata" in kwargs
    assert "trace_id" in kwargs["metadata"]
    assert len(kwargs["metadata"]["trace_id"]) == 32  # 128-bit hex
    assert "span_id" in kwargs["metadata"]
    assert len(kwargs["metadata"]["span_id"]) == 16  # 64-bit hex

    # Cleanup
    otel_trace.set_tracer_provider(otel_trace.NoOpTracerProvider())
