"""Every model in /models must appear in /models/catalog.

V1.6.2 L3 fix: the YAML catalog was frozen at 9 entries while /models
grew to 55 (42 kiki + cloud + vllm + embed). The UI read /models/catalog
and never saw the new models. Catalog must self-fill from /models using
prefix-based default metadata for unknown aliases."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from life_core.api import app


@pytest.mark.asyncio
async def test_catalog_contains_every_model_listed_in_models_endpoint(
    monkeypatch,
):
    flat_models = [
        "kiki-meta-coding",
        "kiki-niche-python",
        "anthropic/claude-sonnet-4-20250514",
        "groq/llama3-70b-8192",
        "openai/gpt-4o",
        "openai/qwen-32b-awq",
        "tei/nomic-embed-text",
    ]

    from life_core import config_api as cfg
    from unittest.mock import AsyncMock, MagicMock

    # Build a fake router that returns the controlled flat_models list
    fake_provider = MagicMock()
    fake_provider.list_models = AsyncMock(return_value=flat_models)

    fake_router = MagicMock()
    fake_router.list_available_providers.return_value = ["litellm"]
    fake_router.providers = {"litellm": fake_provider}

    # Inject directly into models_api (lifespan not triggered by ASGITransport)
    monkeypatch.setattr("life_core.models_api._router", fake_router)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.get("/models/catalog")
    assert r.status_code == 200
    body = r.json()
    catalog_ids = {m["id"] for m in body["models"]}
    for m in flat_models:
        assert m in catalog_ids, f"catalog missing {m}"


@pytest.mark.asyncio
async def test_catalog_entry_has_required_shape(monkeypatch):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.get("/models/catalog")
    body = r.json()
    for m in body["models"]:
        assert "id" in m
        assert "provider" in m
        assert "domain" in m
        assert "description" in m, f"{m['id']} missing description"


def test_default_metadata_for_kiki_niche_prefix():
    from life_core.config_api import default_catalog_entry

    entry = default_catalog_entry("kiki-niche-spice-sim")
    assert entry["provider"] == "kiki-router"
    assert entry["domain"] == "spice-sim"
    assert entry["description"]


def test_default_metadata_for_kiki_meta_prefix():
    from life_core.config_api import default_catalog_entry

    entry = default_catalog_entry("kiki-meta-reasoning")
    assert entry["provider"] == "kiki-router"
    assert entry["domain"] == "reasoning"


def test_default_metadata_for_anthropic_prefix():
    from life_core.config_api import default_catalog_entry

    entry = default_catalog_entry("anthropic/claude-sonnet-4-20250514")
    assert entry["provider"] == "anthropic"
    assert entry["domain"] == "general"


def test_catalog_default_capability_is_chat():
    """Sans hint dans le nom, capability=chat par défaut."""
    from life_core.models_api import _infer_capability

    assert _infer_capability("openai/gpt-4o") == "chat"
    assert _infer_capability("anthropic/claude-sonnet-4-20250514") == "chat"
    assert _infer_capability("kiki-niche-python") == "chat"


def test_catalog_embed_in_name_returns_embedding():
    """Les noms contenant 'embed' ou 'nomic' -> embedding."""
    from life_core.models_api import _infer_capability

    assert _infer_capability("nomic-embed-text") == "embedding"
    assert _infer_capability("openai/text-embedding-3-small") == "embedding"
    assert _infer_capability("BAAI/bge-large-en-v1.5") == "embedding"


def test_catalog_vision_in_name_returns_vision():
    """Les noms contenant 'vision' ou '-vl-' -> vision."""
    from life_core.models_api import _infer_capability

    assert _infer_capability("openai/gpt-4-vision") == "vision"
    assert _infer_capability("qwen-vl-chat") == "vision"


def test_catalog_override_yaml_wins_over_heuristic(tmp_path, monkeypatch):
    """Un override YAML doit primer sur l'heuristique."""
    yaml_content = """
overrides:
  "mystery-model-42": embedding
"""
    cfg = tmp_path / "overrides.yaml"
    cfg.write_text(yaml_content)
    monkeypatch.setenv("F4L_MODELS_OVERRIDES_YAML", str(cfg))

    from life_core.models_api import _classify_capability, _load_overrides
    _load_overrides.cache_clear()
    assert _classify_capability("mystery-model-42") == "embedding"
