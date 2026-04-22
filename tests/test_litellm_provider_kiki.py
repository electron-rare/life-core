"""kiki-router routing tests for LiteLLMProvider."""
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
