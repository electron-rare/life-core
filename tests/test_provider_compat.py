"""Tests for compatibility provider wrappers."""

from __future__ import annotations

from life_core.router.providers.compat_providers import (
    AnthropicProvider,
    GoogleProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
)


def test_compat_provider_ids():
    assert OpenAIProvider().provider_id == "openai"
    assert AnthropicProvider().provider_id == "anthropic"
    assert GoogleProvider().provider_id == "google"
    assert MistralProvider().provider_id == "mistral"
    assert GroqProvider().provider_id == "groq"