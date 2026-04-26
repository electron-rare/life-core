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
