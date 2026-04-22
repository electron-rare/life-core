"""Real LLM call helper for f4l-workflow agents.

Uses an OpenAI-compatible endpoint (mascarade proxy at MASCARADE_URL,
LiteLLM at LITELLM_URL, or life-core's own router). Defaults target
mascarade chat completions with `claude-sonnet-4-6`.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


def _endpoint() -> str:
    for key in ("MASCARADE_URL", "LITELLM_URL", "LIFE_CORE_LLM_URL"):
        value = os.environ.get(key, "").rstrip("/")
        if value:
            return f"{value}/chat/completions"
    return "http://localhost:8000/v1/chat/completions"


def _model() -> str:
    return os.environ.get("F4L_AGENT_MODEL", "claude-sonnet-4-6")


def _api_key() -> str:
    for key in ("MASCARADE_API_KEY", "LITELLM_API_KEY", "ANTHROPIC_API_KEY"):
        value = os.environ.get(key, "")
        if value:
            return value
    return ""


async def call_llm(prompt: str) -> str:
    """POST an OpenAI-compatible chat completion and return the first choice text."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": _model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(_endpoint(), json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected LLM response shape: {e}") from e
