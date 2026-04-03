"""Models API — curated model catalog with domain specializations."""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger("life_core.models_api")

models_router = APIRouter(tags=["Models"])

# Curated catalog of available models with domain info
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
        "id": "ollama/nomic-embed-text",
        "name": "Nomic Embed Text",
        "provider": "ollama",
        "domain": "embeddings",
        "description": "Embeddings pour RAG pipeline (via Cils)",
        "size": "274 MB",
        "location": "Cils",
    },
]

DOMAIN_LABELS = {
    "general": "Général",
    "stm32": "STM32/ARM",
    "kicad": "KiCad/EDA",
    "spice": "SPICE",
    "freecad": "FreeCAD",
    "platformio": "PlatformIO",
    "embedded": "Embarqué",
    "embeddings": "Embeddings",
}


@models_router.get("/models/catalog")
async def model_catalog():
    """Return the curated model catalog with domain metadata."""
    return {
        "models": MODEL_CATALOG,
        "domains": DOMAIN_LABELS,
    }
