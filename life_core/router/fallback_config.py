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

# eu-kiki gateway aliases (electron-server:9300). Priority order encodes
# the desired cascade: heavy reasoning -> medium -> quick -> cloud filet.
_AILIANCE_HEAVY = "ailiance-qwen"       # Qwen3-Next 80B, 262k ctx
_AILIANCE_MEDIUM = "ailiance-granite"   # Granite-4.1 30B
_AILIANCE_QUICK = "ailiance-gemma"      # Gemma 3 4B (Tower)
# _AILIANCE_FR ("ailiance-eurollm") is referenced directly by the eurollm
# entry below; FR-tuned niches (chat-fr, traduction-tech, etc.) still go
# through the legacy kiki-niche-* chain handled above.

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
    # kiki-meta-* / kiki-niche-* keep the original 2-step chain:
    # KXKM llama-server Qwen variant first (closest semantic overlap with
    # Studio MLX base), then cloud cascade. Router tests pin this order.
    for intent in _META_INTENTS:
        m[f"kiki-meta-{intent}"] = [_KXKM_VLLM, _CLOUD_CASCADE]
    for niche in _NICHES:
        m[f"kiki-niche-{niche}"] = [_KXKM_VLLM, _CLOUD_CASCADE]

    # ailiance-* aliases also need fallback chains for when the primary
    # worker (KXKM-AI, Tower, Studio, macM1) is down. Chain peers first,
    # then cloud cascade.
    m["ailiance-qwen"] = [_AILIANCE_MEDIUM, _AILIANCE_QUICK, _CLOUD_CASCADE]
    m["ailiance-granite"] = [_AILIANCE_HEAVY, _AILIANCE_QUICK, _CLOUD_CASCADE]
    m["ailiance-gemma"] = [_AILIANCE_MEDIUM, _AILIANCE_HEAVY, _CLOUD_CASCADE]
    m["ailiance-eurollm"] = [_AILIANCE_HEAVY, _AILIANCE_MEDIUM, _CLOUD_CASCADE]
    m["ailiance-apertus"] = [_AILIANCE_HEAVY, _AILIANCE_MEDIUM, _CLOUD_CASCADE]
    for mlx_alias in (
        "ailiance-mistral",
        "ailiance-ministral",
        "ailiance-ministral-reasoning",
        "ailiance-gemma2",
        "ailiance-gemma4",
    ):
        m[mlx_alias] = [_AILIANCE_QUICK, _AILIANCE_MEDIUM, _AILIANCE_HEAVY, _CLOUD_CASCADE]
    return m


KIKI_TO_VLLM_FALLBACKS: MappingProxyType[str, list[str]] = MappingProxyType(
    _build_fallback_map()
)
