"""LLM Router."""

from __future__ import annotations

import os

import litellm

from life_core.router.providers import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LiteLLMProvider,
)
from life_core.router.router import Router

__all__ = [
    "Router",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LiteLLMProvider",
]

# ---------------------------------------------------------------------------
# LiteLLM custom endpoint registration — dual-stack claude-runner + qwen
#
# claude-runner-sonnet-4-7 is an OpenAI-compat endpoint exposed by the
# claude-runner sidecar (:9300). We register it via litellm's custom
# openai-compatible endpoint mechanism so callers can use the logical
# model name and LiteLLM routes to the right api_base.
#
# Note: litellm.model_list is a flat list of known model *name strings*
# (populated from the bundled model_prices_and_context_window.json).
# Custom endpoint configuration is done via litellm.openai_compatible_endpoints
# or via per-call api_base. We use the openai_compatible_endpoints dict so the
# sidecar URL is resolved once at import time, not per call-site.
#
# The qwen-14b-awq-kxkm entry is owned by LiteLLMProvider(vllm_models=…)
# in api.py. We do NOT duplicate it here.
# ---------------------------------------------------------------------------

_CLAUDE_RUNNER_BASE = os.getenv("CLAUDE_RUNNER_URL", "http://localhost:9300")

# Register the claude-runner sidecar as an OpenAI-compat provider.
# litellm routes "openai/<model>" calls to api_base when the model name
# matches a key here. Idempotent: guard against double-import.
if not hasattr(litellm, "_f4l_claude_runner_registered"):
    litellm.api_base = None  # do not override the global default
    # Store base URL so ClaudeRunnerProvider and direct litellm callers can
    # resolve it from a single source of truth.
    litellm._f4l_claude_runner_base = _CLAUDE_RUNNER_BASE
    litellm._f4l_claude_runner_registered = True

# Declare the auto-fallback chain: claude-runner-sonnet-4-7 → qwen-14b-awq-kxkm.
# Extend the list if fallbacks already contain entries from other modules.
_fallback_entry = {"claude-runner-sonnet-4-7": ["qwen-14b-awq-kxkm"]}
_existing_fallbacks: list = litellm.fallbacks if isinstance(litellm.fallbacks, list) else []
_fallback_keys = {
    next(iter(f)) for f in _existing_fallbacks if isinstance(f, dict)
}
if "claude-runner-sonnet-4-7" not in _fallback_keys:
    litellm.fallbacks = _existing_fallbacks + [_fallback_entry]
