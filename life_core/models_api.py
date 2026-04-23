"""Models API — curated model catalog with domain specializations."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fastapi import APIRouter

from life_core.config_api import default_catalog_entry

if TYPE_CHECKING:
    from life_core.router.router import Router

logger = logging.getLogger("life_core.models_api")

models_router = APIRouter(tags=["Models"])

# Module-level router reference — set during app lifespan via set_models_router()
_router: "Router | None" = None


def set_models_router(r: "Router | None") -> None:
    """Wire the active Router instance so /models/catalog can query it."""
    global _router
    _router = r


_DEFAULT_OVERRIDES = Path(__file__).parent / "config" / "models_overrides.yaml"

_EMBED_HINTS = ("embed", "nomic", "bge-", "gte-", "e5-")
_VISION_HINTS = ("vision", "-vl-", "llava", "pixtral")


def _infer_capability(model_id: str) -> str:
    """Heuristique pure : classe un id de modèle en chat/embedding/vision."""
    low = model_id.lower()
    if any(h in low for h in _EMBED_HINTS):
        return "embedding"
    if any(h in low for h in _VISION_HINTS):
        return "vision"
    return "chat"


@lru_cache(maxsize=1)
def _load_overrides() -> dict[str, str]:
    path = Path(
        os.environ.get("F4L_MODELS_OVERRIDES_YAML", str(_DEFAULT_OVERRIDES))
    )
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("overrides", {}))


def _classify_capability(model_id: str) -> str:
    """Overrides YAML en priorité, sinon heuristique."""
    overrides = _load_overrides()
    if model_id in overrides:
        return overrides[model_id]
    return _infer_capability(model_id)


# Hand-curated catalog entries used as YAML overrides.
# These take precedence over prefix-based defaults from default_catalog_entry().
MODEL_CATALOG = [
    {
        "id": "openai/qwen-32b-awq",
        "name": "Qwen2.5-32B AWQ",
        "provider": "vllm",
        "domain": "general",
        "description": "Modèle principal 14B, AWQ 4-bit, FP8 KV cache + CPU offload",
        "size": "19 GB",
        "location": "KXKM-AI (RTX 4090)",
        "context_window": "128K+ tokens",
    },
    {
        "id": "openai/mascarade-stm32",
        "name": "Mascarade STM32 (LoRA)",
        "provider": "vllm",
        "domain": "stm32",
        "description": "Expert STM32/ARM Cortex-M — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/mascarade-kicad",
        "name": "Mascarade KiCad (LoRA)",
        "provider": "vllm",
        "domain": "kicad",
        "description": "Expert KiCad EDA — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/mascarade-spice",
        "name": "Mascarade SPICE (LoRA)",
        "provider": "vllm",
        "domain": "spice",
        "description": "Expert simulation SPICE — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/mascarade-freecad",
        "name": "Mascarade FreeCAD (LoRA)",
        "provider": "vllm",
        "domain": "freecad",
        "description": "Expert modélisation 3D FreeCAD — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/mascarade-platformio",
        "name": "Mascarade PlatformIO (LoRA)",
        "provider": "vllm",
        "domain": "platformio",
        "description": "Expert PlatformIO/ESP32 — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/mascarade-embedded",
        "name": "Mascarade Embedded (LoRA)",
        "provider": "vllm",
        "domain": "embedded",
        "description": "Expert systèmes embarqués — LoRA adapter sur Qwen 27B",
        "size": "LoRA ~250 MB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "openai/qwen-3b-tower",
        "name": "Qwen2.5-3B Tower",
        "provider": "llama.cpp",
        "domain": "chat",
        "description": "Qwen2.5-3B-Instruct Q4_K_M — llama.cpp local sur Tower",
        "size": "~2GB Q4_K_M",
        "location": "Tower (GPU P2000)",
    },
    {
        "id": "tei/nomic-embed-text",
        "name": "Nomic Embed Text",
        "provider": "tei",
        "domain": "embeddings",
        "description": "Embeddings pour RAG pipeline (TEI, CPU)",
        "size": "274 MB",
        "location": "KXKM-AI (CPU)",
    },
]

DOMAIN_LABELS = {
    "general": "Général",
    "chat": "Chat",
    "stm32": "STM32/ARM",
    "kicad": "KiCad/EDA",
    "spice": "SPICE",
    "freecad": "FreeCAD",
    "platformio": "PlatformIO",
    "embedded": "Embarqué",
    "embeddings": "Embeddings",
}


def _build_yaml_overrides() -> dict[str, dict]:
    """Load hand-curated entries: first from models_catalog.yaml if present,
    else fall back to the in-memory MODEL_CATALOG constant."""
    yaml_path = Path(__file__).parent / "config" / "models_catalog.yaml"
    if yaml_path.exists():
        try:
            import yaml  # type: ignore[import-untyped]
            loaded = yaml.safe_load(yaml_path.read_text()) or {"models": []}
            return {e["id"]: e for e in loaded.get("models", [])}
        except Exception as e:
            logger.warning("Failed to load models_catalog.yaml: %s", e)
    # Fallback to in-memory constant
    return {e["id"]: e for e in MODEL_CATALOG}


@models_router.get("/models/catalog")
async def model_catalog():
    """Return the curated model catalog with domain metadata.

    Merges hand-curated YAML overrides with prefix-based defaults generated
    from the flat /models list. YAML entries win; unknown aliases get a stub
    from default_catalog_entry() so they are never silently dropped.
    """
    yaml_entries = _build_yaml_overrides()

    # Pull flat model IDs from the active router when available
    flat_ids: list[str] = []
    if _router is not None:
        for provider_id in _router.list_available_providers():
            try:
                provider = _router.providers[provider_id]
                flat_ids.extend(await provider.list_models())
            except Exception as e:
                logger.warning("list_models failed for %s: %s", provider_id, e)

    if not flat_ids:
        # No live router — return the static catalog unchanged for backward compat
        static_out = [
            {**entry, "capability": _classify_capability(entry["id"])}
            for entry in MODEL_CATALOG
        ]
        return {
            "models": static_out,
            "domains": DOMAIN_LABELS,
        }

    # Compose: YAML entry wins; else generate defaults
    out: list[dict] = []
    seen: set[str] = set()
    for mid in flat_ids:
        if mid in yaml_entries:
            entry = dict(yaml_entries[mid])
        else:
            entry = dict(default_catalog_entry(mid))
        entry["capability"] = _classify_capability(mid)
        out.append(entry)
        seen.add(mid)

    # Include YAML-only entries (not in /models) as archived
    for mid, entry in yaml_entries.items():
        if mid not in seen:
            archived = dict(entry)
            archived.setdefault("status", "archived")
            archived["capability"] = _classify_capability(mid)
            out.append(archived)

    # Build domains dict: known labels preserved, unknowns get title-cased label
    domains: dict[str, str] = {}
    for m in out:
        d = m.get("domain", "general")
        domains[d] = DOMAIN_LABELS.get(d, d.replace("-", " ").title())

    return {
        "models": out,
        "domains": domains,
    }
