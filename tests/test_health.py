"""Tests for /health aggregation with runtime status."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(monkeypatch):
    """Reset env + import fresh app."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    from life_core.api import app
    return TestClient(app)


def test_health_returns_ok_when_all_providers_healthy(test_client):
    """Avec tous les providers up, status=ok et issues vide."""
    from life_core import api as api_module

    mock_router = MagicMock()
    mock_router.list_available_providers.return_value = ["litellm"]
    mock_router.get_provider_status.return_value = {"litellm": True}

    with patch.object(api_module, "router", mock_router):
        resp = test_client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["issues"] == []
    assert body["router_status"] == {"litellm": True}


def test_health_degraded_when_router_provider_down(test_client):
    """litellm=False dans router_status → status=degraded + issue listée."""
    from life_core import api as api_module

    mock_router = MagicMock()
    mock_router.list_available_providers.return_value = ["litellm"]
    mock_router.get_provider_status.return_value = {"litellm": False}

    with patch.object(api_module, "router", mock_router):
        resp = test_client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "router:litellm:down" in body["issues"]
    assert body["router_status"]["litellm"] is False


def test_health_degraded_when_vllm_ping_fails(test_client, monkeypatch):
    """vllm ping qui throw → status=degraded + issue vllm."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://unreachable:9999")
    from life_core import api as api_module

    mock_router = MagicMock()
    mock_router.list_available_providers.return_value = ["litellm"]
    mock_router.get_provider_status.return_value = {"litellm": True}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connect error"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.object(api_module, "router", mock_router), \
         patch("life_core.api.httpx.AsyncClient", return_value=mock_client):
        resp = test_client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert "backend:vllm:down" in body["issues"]
