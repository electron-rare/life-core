"""Tests for ClaudeRunnerProvider — HTTP adapter for the claude-runner sidecar."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router.providers.claude_runner_provider import ClaudeRunnerProvider
from life_core.router.providers.base import LLMResponse


# ---------------------------------------------------------------------------
# 1. send() happy path — returns LLMResponse with correct shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "life_core.router.providers.claude_runner_provider.httpx.AsyncClient"
)
async def test_send_returns_llm_response_shape(mock_client_cls):
    """send() maps the claude-runner JSON payload into an LLMResponse."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = lambda: {
        "id": "chatcmpl-abc",
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}}
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
        },
        "latency_ms": 1234,
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_http

    provider = ClaudeRunnerProvider(
        base_url="http://localhost:9300", timeout=10
    )
    result = await provider.send(
        [{"role": "user", "content": "hi"}], model="claude-sonnet-4-7"
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "hello"
    assert result.usage["input_tokens"] == 5
    assert result.usage["output_tokens"] == 1
    assert result.metadata["latency_ms"] == 1234
    assert result.provider == "claude-runner"


# ---------------------------------------------------------------------------
# 2. health_check() — returns True on 200 ok
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "life_core.router.providers.claude_runner_provider.httpx.AsyncClient"
)
async def test_health_check_returns_true_on_ok(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = lambda: {"status": "ok"}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_resp)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_http

    provider = ClaudeRunnerProvider(base_url="http://localhost:9300")
    assert await provider.health_check() is True


# ---------------------------------------------------------------------------
# 3. health_check() — returns False on HTTP error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "life_core.router.providers.claude_runner_provider.httpx.AsyncClient"
)
async def test_health_check_returns_false_on_error(mock_client_cls):
    import httpx

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=httpx.HTTPError("conn refused"))
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_http

    provider = ClaudeRunnerProvider(base_url="http://localhost:9300")
    assert await provider.health_check() is False


# ---------------------------------------------------------------------------
# 4. Timeout env var override
# ---------------------------------------------------------------------------


def test_timeout_env_var_override(monkeypatch):
    """CLAUDE_RUNNER_TIMEOUT_S env var overrides the default of 30."""
    monkeypatch.setenv("CLAUDE_RUNNER_TIMEOUT_S", "60")
    provider = ClaudeRunnerProvider(base_url="http://localhost:9300")
    assert provider.timeout == 60


def test_timeout_explicit_arg_wins_over_env(monkeypatch):
    """Explicit timeout= constructor arg beats the env var."""
    monkeypatch.setenv("CLAUDE_RUNNER_TIMEOUT_S", "60")
    provider = ClaudeRunnerProvider(
        base_url="http://localhost:9300", timeout=10
    )
    assert provider.timeout == 10


def test_timeout_default_is_30():
    """Default timeout is 30s (J1 validation: warm calls 12-20s, cold up to 90s)."""
    import os
    os.environ.pop("CLAUDE_RUNNER_TIMEOUT_S", None)
    provider = ClaudeRunnerProvider(base_url="http://localhost:9300")
    assert provider.timeout == 30


# ---------------------------------------------------------------------------
# 5. send() retries on 503 — succeeds on 3rd attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "life_core.router.providers.claude_runner_provider.httpx.AsyncClient"
)
@patch(
    "life_core.router.providers.claude_runner_provider.asyncio.sleep",
    new_callable=AsyncMock,
)
async def test_retries_on_503_succeeds_third_call(mock_sleep, mock_client_cls):
    """send() retries on 503 and returns LLMResponse on the third call."""
    import httpx

    bad_resp = MagicMock()
    bad_resp.status_code = 503
    bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=MagicMock(),
        response=bad_resp,
    )

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status = MagicMock()
    good_resp.json = lambda: {
        "id": "x",
        "choices": [{"message": {"role": "assistant", "content": "y"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "latency_ms": 100,
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=[bad_resp, bad_resp, good_resp])
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_http

    provider = ClaudeRunnerProvider(
        base_url="http://localhost:9300", max_retries=3, timeout=5
    )
    result = await provider.send([{"role": "user", "content": "hi"}])

    assert result.content == "y"
    assert mock_http.post.call_count == 3
    assert mock_sleep.call_count == 2  # slept between attempts 1→2 and 2→3


# ---------------------------------------------------------------------------
# 6. OTEL span emitted on send()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "life_core.router.providers.claude_runner_provider.httpx.AsyncClient"
)
async def test_otel_span_emitted(mock_client_cls):
    """send() emits a span named 'claude_runner.send'."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace as otel_trace
    import life_core.router.providers.claude_runner_provider as provider_mod

    exporter = InMemorySpanExporter()
    sdk_provider = TracerProvider()
    sdk_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Temporarily swap the module-level tracer to use our SDK provider
    original_tracer = provider_mod.tracer
    provider_mod.tracer = sdk_provider.get_tracer("life_core.providers.claude_runner")

    try:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = lambda: {
            "id": "chatcmpl-span",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "latency_ms": 42,
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_http

        provider = ClaudeRunnerProvider(base_url="http://localhost:9300", timeout=5)
        result = await provider.send([{"role": "user", "content": "hi"}])

        assert result.content == "ok"

        finished = exporter.get_finished_spans()
        span_names = [s.name for s in finished]
        assert "claude_runner.send" in span_names
    finally:
        provider_mod.tracer = original_tracer
