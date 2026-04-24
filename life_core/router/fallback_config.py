"""Static fallback map for Studio kiki-* aliases.

When the Studio MLX runtime is unreachable (tunnel down, Studio
rebooted, kernel panic) the LiteLLM router can retry the request
against the KXKM llama-server Qwen3.6-35B-Q4 variant, which runs
the same base weights at slightly lower fidelity and acts as a
stand-in.

A final cloud fallback (Claude Sonnet 4) ensures the request
never fails outright — the user gets a cloud response instead
of a 5xx.
"""
from __future__ import annotations

from types import MappingProxyType


_KXKM_VLLM = "openai/qwen3.6-35b-kxkm"
_CLOUD_CASCADE = "anthropic/claude-sonnet-4-20250514"

_META_INTENTS = (
    "agentic",
    "coding",
    "creative",
    "quick-reply",
    "reasoning",
    "research",
    "tool-use",
)

_NICHES = (
    "chat-fr", "components", "cpp", "devops", "docker", "dsp",
    "electronics", "embedded", "emc", "freecad", "html-css", "iot",
    "kicad-dsl", "kicad-pcb", "llm-ops", "llm-orch", "lua-upy",
    "math", "ml-training", "music-audio", "platformio", "power",
    "python", "reasoning", "rust", "security", "shell", "spice",
    "spice-sim", "sql", "stm32", "typescript", "web-backend",
    "web-frontend", "yaml-json",
)


def _build_fallback_map() -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for intent in _META_INTENTS:
        m[f"kiki-meta-{intent}"] = [_KXKM_VLLM, _CLOUD_CASCADE]
    for niche in _NICHES:
        m[f"kiki-niche-{niche}"] = [_KXKM_VLLM, _CLOUD_CASCADE]
    return m


KIKI_TO_VLLM_FALLBACKS: MappingProxyType[str, list[str]] = MappingProxyType(
    _build_fallback_map()
)
