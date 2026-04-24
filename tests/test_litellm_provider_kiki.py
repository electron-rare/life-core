"""kiki-router routing tests for LiteLLMProvider."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from life_core.router.providers.litellm_provider import LiteLLMProvider


def test_kiki_model_resolves_to_openai_prefix():
    p = LiteLLMProvider(
        models=["kiki-meta-coding", "kiki-niche-stm32"],
        kiki_full_base_url="http://host.docker.internal:9200/v1",
        kiki_full_models={"kiki-meta-coding", "kiki-niche-stm32"},
    )
    assert p._resolve_model_name("kiki-meta-coding") == "openai/kiki-meta-coding"
    assert p._resolve_model_name("kiki-niche-stm32") == "openai/kiki-niche-stm32"


def test_kiki_kwargs_override_api_base():
    p = LiteLLMProvider(
        models=["kiki-meta-coding"],
        kiki_full_base_url="http://host.docker.internal:9200/v1",
        kiki_full_models={"kiki-meta-coding"},
    )
    kw = p._build_call_kwargs("openai/kiki-meta-coding", {})
    assert kw["api_base"] == "http://host.docker.internal:9200/v1"


def test_non_kiki_openai_model_not_routed_to_kiki_base():
    p = LiteLLMProvider(
        models=["openai/gpt-4o", "kiki-meta-coding"],
        kiki_full_base_url="http://host.docker.internal:9200/v1",
        kiki_full_models={"kiki-meta-coding"},
    )
    kw = p._build_call_kwargs("openai/gpt-4o", {})
    assert kw.get("api_base") is None


def test_kiki_without_base_url_raises():
    with pytest.raises(ValueError, match="kiki"):
        LiteLLMProvider(
            models=["kiki-meta-coding"],
            kiki_full_models={"kiki-meta-coding"},
        )


def test_kiki_empty_set_no_error():
    # No kiki models → no constraint violation
    p = LiteLLMProvider(models=["claude-sonnet"])
    assert p._resolve_model_name("claude-sonnet") == "claude-sonnet"


def _mock_litellm_response(model="openai/kiki-niche-python"):
    usage = MagicMock()
    usage.prompt_tokens = 3
    usage.completion_tokens = 4
    message = MagicMock()
    message.content = "ok"
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_send_triggers_pre_routing_and_downgrades_meta():
    """send() calls kiki pre-routing and forwards the downgraded model to litellm."""
    provider = LiteLLMProvider(
        models=["kiki-meta-coding", "kiki-niche-python"],
        kiki_full_base_url="http://kiki:9200/v1",
        kiki_full_models={"kiki-meta-coding", "kiki-niche-python"},
    )

    with patch(
        "life_core.router.providers.litellm_provider._kiki_resolve_model",
        new_callable=AsyncMock,
    ) as mock_resolve, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_call:
        mock_resolve.return_value = "kiki-niche-python"
        mock_call.return_value = _mock_litellm_response()
        result = await provider.send(
            messages=[{"role": "user", "content": "écris une fonction python"}],
            model="kiki-meta-coding",
        )

    mock_resolve.assert_awaited_once()
    args, kwargs = mock_resolve.call_args
    assert args[0] == "kiki-meta-coding"
    assert kwargs["kiki_full_base_url"] == "http://kiki:9200/v1"
    _, call_kwargs = mock_call.call_args
    assert call_kwargs["model"] == "openai/kiki-niche-python"
    assert result.model == "openai/kiki-niche-python"


@pytest.mark.asyncio
async def test_send_keeps_model_when_pre_routing_returns_original():
    """Score below threshold -> pre-routing returns original, meta passes through."""
    provider = LiteLLMProvider(
        models=["kiki-meta-coding"],
        kiki_full_base_url="http://kiki:9200/v1",
        kiki_full_models={"kiki-meta-coding"},
    )

    with patch(
        "life_core.router.providers.litellm_provider._kiki_resolve_model",
        new_callable=AsyncMock,
    ) as mock_resolve, patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_call:
        mock_resolve.return_value = "kiki-meta-coding"
        mock_call.return_value = _mock_litellm_response(
            model="openai/kiki-meta-coding"
        )
        await provider.send(
            messages=[{"role": "user", "content": "bonjour"}],
            model="kiki-meta-coding",
        )

    _, call_kwargs = mock_call.call_args
    assert call_kwargs["model"] == "openai/kiki-meta-coding"


@pytest.mark.asyncio
async def test_send_non_kiki_model_pre_routing_passthrough():
    """Non-kiki models pass through unchanged even with pre-routing wired."""
    provider = LiteLLMProvider(models=["openai/gpt-4o"])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_litellm_response(model="openai/gpt-4o")
        await provider.send(
            messages=[{"role": "user", "content": "hello"}],
            model="openai/gpt-4o",
        )

    _, call_kwargs = mock_call.call_args
    assert call_kwargs["model"] == "openai/gpt-4o"
