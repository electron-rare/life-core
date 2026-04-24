"""Router failover: kiki-* models cascade to the KXKM llama-server
vllm variant when Studio is unreachable."""
from __future__ import annotations

import pytest

from life_core.router.fallback_config import KIKI_TO_VLLM_FALLBACKS


def test_fallback_map_covers_every_kiki_alias():
    """Every kiki-meta-* and kiki-niche-* must have at least one
    fallback target so meta-mode routing has a degraded path."""
    for alias, targets in KIKI_TO_VLLM_FALLBACKS.items():
        assert alias.startswith("kiki-"), f"{alias} is not a kiki alias"
        assert len(targets) >= 1, f"{alias} has empty fallback list"
        for t in targets:
            assert t.startswith("openai/") or t.startswith("anthropic/"), (
                f"{alias} -> {t}: fallback must be a namespaced provider"
            )


def test_fallback_first_target_is_kxkm_qwen():
    """Primary fallback for a kiki alias should be the KXKM llama-server
    Qwen Q4 variant, which has the closest semantic overlap with the
    Studio MLX base before any LoRA."""
    expected = "openai/qwen3.6-35b-kxkm"
    sampled = ["kiki-niche-python", "kiki-meta-coding", "kiki-niche-chat-fr"]
    for alias in sampled:
        assert KIKI_TO_VLLM_FALLBACKS[alias][0] == expected, (
            f"{alias} primary fallback should be {expected}"
        )


def test_fallback_map_is_frozen():
    """Ensure the export is a Mapping, not a mutable global that code
    elsewhere can monkey-patch at runtime."""
    import types
    assert isinstance(KIKI_TO_VLLM_FALLBACKS, types.MappingProxyType), (
        "KIKI_TO_VLLM_FALLBACKS must be a MappingProxyType to prevent "
        "runtime mutation"
    )


@pytest.mark.asyncio
async def test_kiki_failure_cascades_to_cloud(monkeypatch):
    """Verify that the early-return path for kiki-* models injects the
    static fallback chain as the fallbacks= kwarg into provider.send().
    The provider (LiteLLM SDK) then owns the model-swap cascade.
    We assert the kwarg is set correctly and the response is returned."""
    from life_core.router.router import Router
    from life_core.router.providers.base import LLMResponse

    received_kwargs: dict = {}

    class MockProvider:
        provider_id = "litellm"
        is_available = True

        async def send(self, messages, model, **kwargs):
            received_kwargs.update(kwargs)
            # Simulate LiteLLM SDK successfully routing to the cloud
            # fallback after the Studio/KXKM endpoints are exhausted.
            return LLMResponse(
                content="cloud ok",
                model=model,
                provider="litellm",
                usage={"input_tokens": 1, "output_tokens": 2},
            )

        async def stream(self, messages, model, **kwargs):
            raise NotImplementedError

        async def health_check(self):
            return True

    router = Router()
    router.register_provider(MockProvider(), is_primary=True)

    response = await router.send(
        messages=[{"role": "user", "content": "hi"}],
        model="kiki-niche-python",
    )

    assert response.content == "cloud ok"
    # Confirm the fallback chain was injected into the provider call.
    assert "fallbacks" in received_kwargs, (
        "fallbacks= kwarg must be injected for kiki-* models"
    )
    expected_primary = "openai/qwen3.6-35b-kxkm"
    assert received_kwargs["fallbacks"][0] == expected_primary, (
        f"First fallback must be KXKM vllm ({expected_primary})"
    )
    assert received_kwargs["fallbacks"][1] == "anthropic/claude-sonnet-4-20250514", (
        "Second fallback must be the cloud cascade"
    )
