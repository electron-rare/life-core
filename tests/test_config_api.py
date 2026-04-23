"""Tests for Configuration API router."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.config_api import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── GET /config/providers ──────────────────────────────────────────────────────

def test_list_providers_returns_all_known(client):
    """Should return all 7 known providers."""
    with patch("life_core.config_api._get_redis", return_value=AsyncMock(return_value=None)):
        with patch("life_core.config_api._get_provider_key", new=AsyncMock(return_value=(None, "unconfigured"))):
            with patch("life_core.config_api._get_provider_meta", new=AsyncMock(return_value={})):
                resp = client.get("/config/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    names = [p["name"] for p in data]
    for name in ("anthropic", "openai", "mistral", "groq", "google", "ollama", "vllm"):
        assert name in names


def test_list_providers_masks_key(client):
    """Provider with key should return masked key, not the raw value."""
    async def fake_get_key(name):
        if name == "anthropic":
            return "sk-ant-api03-ABCDEF1234567890", "env"
        return None, "unconfigured"

    async def fake_get_meta(_name):
        return {}

    with patch("life_core.config_api._get_provider_key", side_effect=fake_get_key):
        with patch("life_core.config_api._get_provider_meta", side_effect=fake_get_meta):
            resp = client.get("/config/providers")
    assert resp.status_code == 200
    data = resp.json()
    anthropic = next(p for p in data if p["name"] == "anthropic")
    assert anthropic["source"] == "env"
    assert anthropic["masked_key"] is not None
    assert "ABCDEF" not in anthropic["masked_key"]
    assert "***" in anthropic["masked_key"]


# ── PUT /config/providers/{name} ──────────────────────────────────────────────

def test_update_provider_unknown_returns_404(client):
    resp = client.put("/config/providers/unknown_provider", json={"active": True})
    assert resp.status_code == 404


def test_update_provider_stores_in_redis(client):
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()

    async def fake_get_key(name):
        return "sk-newkey1234567890", "redis"

    async def fake_get_meta(_name):
        return {"active": True, "priority": 0}

    with patch("life_core.config_api._get_redis", new=AsyncMock(return_value=mock_redis)):
        with patch("life_core.config_api._get_provider_key", side_effect=fake_get_key):
            with patch("life_core.config_api._get_provider_meta", side_effect=fake_get_meta):
                resp = client.put("/config/providers/openai", json={"api_key": "sk-newkey1234567890", "active": True, "priority": 1})

    assert resp.status_code == 200
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    stored = json.loads(call_args[0][1])
    assert stored["api_key"] == "sk-newkey1234567890"
    assert stored["active"] is True
    assert stored["priority"] == 1


# ── POST /config/providers/{name}/test ────────────────────────────────────────

def test_test_provider_no_key_returns_not_ok(client):
    async def fake_get_key(name):
        return None, "unconfigured"

    with patch("life_core.config_api._get_provider_key", side_effect=fake_get_key):
        resp = client.post("/config/providers/anthropic/test")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "key" in data["error"].lower()


def test_test_provider_unknown_returns_404(client):
    resp = client.post("/config/providers/no_such_provider/test")
    assert resp.status_code == 404


# ── GET /config/platform ──────────────────────────────────────────────────────

def test_platform_health_returns_services(client):
    import httpx as real_httpx

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {"models": []}

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with patch("life_core.config_api._get_redis", new=AsyncMock(return_value=mock_redis)):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=False)
            mock_async_client.get = AsyncMock(return_value=mock_response_ok)
            mock_client_class.return_value = mock_async_client
            resp = client.get("/config/platform")

    assert resp.status_code == 200
    data = resp.json()
    assert "services" in data
    names = [s["name"] for s in data["services"]]
    for svc in ("redis", "qdrant", "ollama"):
        assert svc in names


def test_platform_exposes_ui_feature_flags(client, monkeypatch):
    """F4L_UI_FEATURE_X=false should be reported in /config/platform."""
    monkeypatch.setenv("F4L_UI_FEATURE_GOVERNANCE", "false")
    monkeypatch.setenv("F4L_UI_FEATURE_DATASHEETS", "false")
    monkeypatch.setenv("F4L_UI_FEATURE_CHAT", "true")

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {"models": []}

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with patch("life_core.config_api._get_redis", new=AsyncMock(return_value=mock_redis)):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_async_client = AsyncMock()
            mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
            mock_async_client.__aexit__ = AsyncMock(return_value=False)
            mock_async_client.get = AsyncMock(return_value=mock_response_ok)
            mock_client_class.return_value = mock_async_client
            resp = client.get("/config/platform")

    assert resp.status_code == 200
    body = resp.json()
    assert "ui_features" in body
    flags = body["ui_features"]
    assert flags["governance"] is False
    assert flags["datasheets"] is False
    assert flags["chat"] is True
    # Default true when env var is not set
    assert flags["dashboard"] is True


# ── GET /config/preferences ───────────────────────────────────────────────────

def test_get_preferences_returns_defaults_when_empty(client):
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    with patch("life_core.config_api._get_redis", new=AsyncMock(return_value=mock_redis)):
        resp = client.get("/config/preferences")

    assert resp.status_code == 200
    data = resp.json()
    assert data["rag_enabled"] is False
    assert data["language"] == "FR"


# ── PUT /config/preferences ───────────────────────────────────────────────────

def test_save_preferences_persists_to_redis(client):
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock()

    payload = {"default_model": "openai/gpt-4o", "rag_enabled": True, "language": "EN"}

    with patch("life_core.config_api._get_redis", new=AsyncMock(return_value=mock_redis)):
        resp = client.put("/config/preferences", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["default_model"] == "openai/gpt-4o"
    assert data["rag_enabled"] is True
    assert data["language"] == "EN"
    mock_redis.set.assert_called_once()
