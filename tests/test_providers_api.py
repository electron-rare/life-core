"""Tests for GET /api/providers (list of LLM providers + status)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI


def test_providers_returns_list_with_status():
    """Chaque provider renvoyé a id, name, status, models_count."""
    from life_core.providers_api import providers_router
    from life_core import providers_api

    mock_router = MagicMock()
    mock_router.list_available_providers.return_value = ["litellm"]
    mock_router.get_provider_status.return_value = {"litellm": True}

    async def fake_list_models(self):
        return ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]

    mock_provider = MagicMock()
    mock_provider.list_models = fake_list_models.__get__(mock_provider)
    mock_router.providers = {"litellm": mock_provider}

    app = FastAPI()
    app.include_router(providers_router)
    client = TestClient(app)

    with patch.object(providers_api, "_get_router", return_value=mock_router):
        resp = client.get("/api/providers")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["providers"], list)
    assert len(body["providers"]) == 1
    p = body["providers"][0]
    assert p["id"] == "litellm"
    assert p["status"] == "up"
    assert p["models_count"] == 2


def test_providers_reports_down_status():
    """Un provider avec get_provider_status=False → status='down'."""
    from life_core.providers_api import providers_router
    from life_core import providers_api

    mock_router = MagicMock()
    mock_router.list_available_providers.return_value = ["litellm"]
    mock_router.get_provider_status.return_value = {"litellm": False}
    mock_provider = MagicMock()

    async def fake_list_models(self):
        return []

    mock_provider.list_models = fake_list_models.__get__(mock_provider)
    mock_router.providers = {"litellm": mock_provider}

    app = FastAPI()
    app.include_router(providers_router)
    client = TestClient(app)

    with patch.object(providers_api, "_get_router", return_value=mock_router):
        body = client.get("/api/providers").json()
    assert body["providers"][0]["status"] == "down"
