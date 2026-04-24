"""Tests for vLLM routing in LiteLLMProvider."""
import pytest
from unittest.mock import AsyncMock, patch

from life_core.router.providers.litellm_provider import LiteLLMProvider


def test_build_call_kwargs_routes_vllm_model():
    """vLLM models get api_base injected."""
    provider = LiteLLMProvider(
        models=["openai/qwen-27b-awq", "ollama/llama3"],
        ollama_api_base="http://cils:11434",
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq", "openai/mascarade-stm32"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/qwen-27b-awq", {})
    assert kwargs["api_base"] == "http://kxkm:8000"


def test_build_call_kwargs_routes_ollama_model():
    """Ollama models still get ollama api_base."""
    provider = LiteLLMProvider(
        models=["openai/qwen-27b-awq", "ollama/llama3"],
        ollama_api_base="http://cils:11434",
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("ollama/llama3", {})
    assert kwargs["api_base"] == "http://cils:11434"


def test_build_call_kwargs_cloud_model_no_api_base():
    """Cloud models get no api_base override."""
    provider = LiteLLMProvider(
        models=["anthropic/claude-sonnet-4-20250514"],
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/qwen-27b-awq"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("anthropic/claude-sonnet-4-20250514", {})
    assert "api_base" not in kwargs


def test_build_call_kwargs_vllm_lora_model():
    """LoRA model names route to vLLM."""
    provider = LiteLLMProvider(
        models=["openai/mascarade-stm32"],
        vllm_api_base="http://kxkm:8000",
        vllm_models={"openai/mascarade-stm32"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/mascarade-stm32", {})
    assert kwargs["api_base"] == "http://kxkm:8000"


def test_vllm_defaults_to_empty():
    """No vLLM config = no vLLM routing."""
    provider = LiteLLMProvider(models=["openai/gpt-4o"])
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/gpt-4o", {})
    assert "api_base" not in kwargs


import os


def test_api_loads_vllm_env_vars():
    """api.py should read VLLM_BASE_URL and VLLM_MODELS and pass to provider."""
    env = {
        "VLLM_BASE_URL": "http://localhost:11436",
        "VLLM_MODELS": "openai/qwen-27b-awq,openai/mascarade-stm32",
        "OPENAI_API_KEY": "sk-test",
    }
    with patch.dict(os.environ, env, clear=False):
        vllm_base = os.getenv("VLLM_BASE_URL")
        vllm_models_str = os.getenv("VLLM_MODELS", "")
        vllm_models = set()
        if vllm_base and vllm_models_str:
            vllm_models = {m.strip() for m in vllm_models_str.split(",")}

        assert vllm_base == "http://localhost:11436"
        assert vllm_models == {"openai/qwen-27b-awq", "openai/mascarade-stm32"}

        provider = LiteLLMProvider(
            models=list(vllm_models),
            vllm_api_base=vllm_base,
            vllm_models=vllm_models,
        )
        assert provider.vllm_api_base == "http://localhost:11436"
        assert "openai/qwen-27b-awq" in provider.vllm_models


def test_vllm_models_use_vllm_api_key_not_openai(monkeypatch):
    """When a model is in vllm_models, api_key sent must be VLLM_API_KEY, not OPENAI_API_KEY.

    Regression guard: OPENAI_API_BASE leaked globally used to redirect every
    openai/* call to local vLLM with the real sk-proj-... key. vLLM returned
    401, LiteLLM marked the pool down, Dashboard flipped to "Degraded". Fix:
    inject a dedicated VLLM_API_KEY per call so the real OpenAI key never
    reaches the local pool.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-public-openai-key")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-secret-key")

    provider = LiteLLMProvider(
        models=["openai/qwen-32b-awq-kxkm"],
        vllm_api_base="http://test-vllm:8002/v1",
        vllm_models={"openai/qwen-32b-awq-kxkm"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/qwen-32b-awq-kxkm", {})

    assert kwargs.get("api_base") == "http://test-vllm:8002/v1"
    assert kwargs.get("api_key") == "vllm-secret-key", (
        f"Expected vllm-secret-key but got {kwargs.get('api_key')}"
    )
    assert "sk-proj" not in (kwargs.get("api_key") or "")


def test_vllm_api_key_default_when_env_unset(monkeypatch):
    """With VLLM_API_KEY unset, fall back to the shared local token default."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-public-openai-key")

    provider = LiteLLMProvider(
        models=["openai/qwen-32b-awq-kxkm"],
        vllm_api_base="http://test-vllm:8002/v1",
        vllm_models={"openai/qwen-32b-awq-kxkm"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/qwen-32b-awq-kxkm", {})

    assert kwargs.get("api_key") == "vllm-er-2026"
    assert "sk-proj" not in (kwargs.get("api_key") or "")


def test_non_vllm_openai_model_keeps_openai_key(monkeypatch):
    """openai/* models outside vllm_models must NOT have VLLM_API_KEY injected.

    LiteLLM default: reads OPENAI_API_KEY from env — we should not override
    kwargs for cloud-bound openai/* models. Ensures cloud fallback stays
    functional once the global OPENAI_API_BASE is removed from compose.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-public-openai-key")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-secret-key")

    provider = LiteLLMProvider(
        models=["openai/gpt-4o-mini"],
        vllm_api_base="http://test-vllm:8002/v1",
        vllm_models={"openai/qwen-32b-awq-kxkm"},
    )
    with patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        kwargs = provider._build_call_kwargs("openai/gpt-4o-mini", {})

    assert kwargs.get("api_key") != "vllm-secret-key"
    assert "api_base" not in kwargs


def test_health_check_prefers_vllm_over_cloud_model(monkeypatch):
    """health_check doit pinger un vLLM local quand dispo, pas le premier model cloud."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-key-will-fail-cloud")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-secret-key")

    provider = LiteLLMProvider(
        models=["openai/gpt-4o", "openai/qwen-32b-awq-kxkm"],
        vllm_api_base="http://test-vllm:8002/v1",
        vllm_models={"openai/qwen-32b-awq-kxkm"},
    )

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class Resp:
            choices = [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": "pong"})()},
                )()
            ]

        return Resp()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        side_effect=fake_acompletion,
    ), patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        import asyncio
        result = asyncio.run(provider.health_check())

    assert result is True
    assert captured.get("model") == "openai/qwen-32b-awq-kxkm", (
        f"health_check pinged {captured.get('model')} instead of vLLM model"
    )
    assert captured.get("api_key") == "vllm-secret-key"


def test_health_check_prefers_kiki_when_no_vllm(monkeypatch):
    """Sans vLLM dispo, fallback sur kiki-router."""
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-key")
    monkeypatch.setenv("KIKI_FULL_API_KEY", "kiki-local-key")

    provider = LiteLLMProvider(
        models=["openai/gpt-4o", "kiki-niche-python"],
        kiki_full_base_url="http://test-kiki:9200/v1",
        kiki_full_models={"kiki-niche-python"},
    )

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class Resp:
            choices = [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": "pong"})()},
                )()
            ]

        return Resp()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        side_effect=fake_acompletion,
    ), patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        import asyncio
        result = asyncio.run(provider.health_check())

    assert result is True
    assert "kiki" in (captured.get("model") or "")


def test_health_check_falls_back_to_first_model_if_no_local(monkeypatch):
    """Si aucun pool local, fallback sur self.models[0] (comportement actuel)."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-key")

    provider = LiteLLMProvider(models=["openai/gpt-4o", "anthropic/claude"])

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        class Resp:
            choices = [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": "pong"})()},
                )()
            ]

        return Resp()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        side_effect=fake_acompletion,
    ), patch("opentelemetry.trace.get_current_span") as mock_span:
        mock_span.return_value.get_span_context.return_value.trace_id = 0
        import asyncio
        result = asyncio.run(provider.health_check())

    assert result is True
    assert captured.get("model") == "openai/gpt-4o"
