"""Guardrails against model name collisions across providers."""
from __future__ import annotations

import os


KIKI_PREFIX = "kiki-"
EXPECTED_HOSTS = ("kxkm", "studio")


def test_vllm_models_have_host_suffix_when_openai_prefixed():
    """Any openai/* entry in VLLM_MODELS that is not a catalog
    cloud model (gpt-*, text-*, whisper-*, dall-e-*) must end
    with a known host suffix to avoid kiki/vllm/cloud collisions.
    V1.6 C1 renamed the 9 vllm-hosted variants; this test keeps
    a future commit from regressing that."""
    raw = os.environ.get("VLLM_MODELS", "")
    if not raw:
        return  # env not loaded in this pytest context

    items = [m.strip() for m in raw.split(",") if m.strip()]
    cloud_prefixes = (
        "openai/gpt-",
        "openai/text-",
        "openai/whisper-",
        "openai/dall-e-",
        "openai/tts-",
    )

    for item in items:
        if not item.startswith("openai/"):
            continue
        if item.startswith(cloud_prefixes):
            continue
        # Locally-hosted openai/* must have a host suffix.
        assert any(
            item.endswith(f"-{h}") for h in EXPECTED_HOSTS
        ), (
            f"VLLM_MODELS entry {item!r} lacks a host suffix "
            f"(expected one of {EXPECTED_HOSTS})"
        )


def test_kiki_prefix_not_used_in_vllm_models():
    """kiki-* is reserved for the kiki-router; vllm must never
    ship a model whose name starts with that prefix."""
    raw = os.environ.get("VLLM_MODELS", "")
    for m in raw.split(","):
        m = m.strip()
        if not m:
            continue
        assert not m.removeprefix("openai/").startswith(KIKI_PREFIX), (
            f"VLLM_MODELS entry {m!r} collides with the kiki-* namespace"
        )


def test_hygiene_test_fires_on_bad_env(monkeypatch):
    """Self-test: confirm the assertion actually bites when an
    unsuffixed openai/* entry appears in VLLM_MODELS."""
    monkeypatch.setenv("VLLM_MODELS", "openai/qwen3.6-35b,openai/gpt-4o")
    try:
        test_vllm_models_have_host_suffix_when_openai_prefixed()
    except AssertionError as e:
        assert "qwen3.6-35b" in str(e)
        return
    raise AssertionError("test did not catch the bad VLLM_MODELS value")
