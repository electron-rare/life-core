"""Models API — curated model catalog with domain specializations."""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger("life_core.models_api")

models_router = APIRouter(tags=["Models"])

# Curated catalog of available models with domain info
MODEL_CATALOG = [
    {
        "id": "qwen3:4b",
        "name": "Qwen3 4B",
        "provider": "ollama",
        "domain": "general",
        "description": "Modèle général léger, rapide",
        "size": "2.5 GB",
        "location": "Tower",
    },
    {
        "id": "qwen35-opus-distilled",
        "name": "Qwen3.5-27B Opus Distilled",
        "provider": "ollama-gpu",
        "domain": "reasoning",
        "description": "Raisonnement avancé, distillé depuis Claude 4.6 Opus",
        "size": "16 GB",
        "location": "KXKM-AI (RTX 4090)",
    },
    {
        "id": "codestral:latest",
        "name": "Codestral",
        "provider": "ollama-gpu",
        "domain": "code",
        "description": "Génération de code (Mistral)",
        "size": "12 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "devstral:latest",
        "name": "Devstral",
        "provider": "ollama-gpu",
        "domain": "code",
        "description": "Développement logiciel (Mistral)",
        "size": "14 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-stm32:latest",
        "name": "Mascarade STM32",
        "provider": "ollama-gpu",
        "domain": "stm32",
        "description": "Expert STM32/ARM Cortex-M (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-kicad:latest",
        "name": "Mascarade KiCad",
        "provider": "ollama-gpu",
        "domain": "kicad",
        "description": "Expert KiCad EDA (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-spice:latest",
        "name": "Mascarade SPICE",
        "provider": "ollama-gpu",
        "domain": "spice",
        "description": "Expert simulation SPICE (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-freecad:latest",
        "name": "Mascarade FreeCAD",
        "provider": "ollama-gpu",
        "domain": "freecad",
        "description": "Expert modélisation 3D FreeCAD (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-platformio:latest",
        "name": "Mascarade PlatformIO",
        "provider": "ollama-gpu",
        "domain": "platformio",
        "description": "Expert PlatformIO/ESP32 (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
    {
        "id": "mascarade-embedded:latest",
        "name": "Mascarade Embedded",
        "provider": "ollama-gpu",
        "domain": "embedded",
        "description": "Expert systèmes embarqués (fine-tuné domaine)",
        "size": "2.5 GB",
        "location": "KXKM-AI",
    },
]

DOMAIN_LABELS = {
    "general": "Général",
    "reasoning": "Raisonnement",
    "code": "Code",
    "stm32": "STM32/ARM",
    "kicad": "KiCad/EDA",
    "spice": "SPICE",
    "freecad": "FreeCAD",
    "platformio": "PlatformIO",
    "embedded": "Embarqué",
}


@models_router.get("/models/catalog")
async def model_catalog():
    """Return the curated model catalog with domain metadata."""
    return {
        "models": MODEL_CATALOG,
        "domains": DOMAIN_LABELS,
    }
