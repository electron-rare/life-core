"""LLM Router."""

from __future__ import annotations

import os

from life_core.router.providers import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LiteLLMProvider,
)
from life_core.router.router import Router


def build_backend_url(path: str) -> str:
    """Build an absolute URL to the default OpenAI-compat backend.

    Resolution order: ``VLLM_BASE_URL`` (KXKM MLX router via tunnel)
    → ``LOCAL_LLM_URL`` (Tower llama.cpp) → localhost fallback. The
    returned URL includes the requested path, joined safely.
    """
    base = (
        os.getenv("VLLM_BASE_URL")
        or os.getenv("LOCAL_LLM_URL")
        or "http://localhost:8000"
    )
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def backend_headers() -> dict[str, str]:
    """Return auth + content headers for OpenAI-compat backend calls.

    Uses ``VLLM_API_KEY`` when set (MLX router often runs with a
    shared token), else falls back to ``OPENAI_API_KEY`` so the same
    header shape works against real OpenAI too.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


__all__ = [
    "Router",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LiteLLMProvider",
    "build_backend_url",
    "backend_headers",
]
