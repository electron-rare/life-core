"""Compatibility provider wrappers built on top of LiteLLM.

These wrappers keep explicit provider identities while reusing a unified
LiteLLM execution backend.
"""

from __future__ import annotations

from life_core.router.providers.litellm_provider import LiteLLMProvider


class OpenAIProvider(LiteLLMProvider):
    def __init__(self, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or ["openai/gpt-4o-mini"], **kwargs)
        self.provider_id = "openai"


class AnthropicProvider(LiteLLMProvider):
    def __init__(self, models: list[str] | None = None, **kwargs):
        super().__init__(
            models=models
            or [
                "anthropic/claude-sonnet-4-20250514",
                "anthropic/claude-3-5-haiku-latest",
            ],
            **kwargs,
        )
        self.provider_id = "anthropic"


class GoogleProvider(LiteLLMProvider):
    def __init__(self, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or ["gemini/gemini-1.5-flash"], **kwargs)
        self.provider_id = "google"


class MistralProvider(LiteLLMProvider):
    def __init__(self, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or ["mistral/mistral-small-latest"], **kwargs)
        self.provider_id = "mistral"


class GroqProvider(LiteLLMProvider):
    def __init__(self, models: list[str] | None = None, **kwargs):
        super().__init__(models=models or ["groq/llama-3.1-8b-instant"], **kwargs)
        self.provider_id = "groq"