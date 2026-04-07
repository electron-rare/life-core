"""Tests for browser_runner_api.py — health and scrape endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from life_core.browser_runner_api import app
from life_core.services.browser import (
    BrowserDependencyMissingError,
    BrowserRemoteRunnerError,
    BrowserServiceError,
)


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "browser-runner"


# ---------------------------------------------------------------------------
# /scrape — success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_success(client):
    mock_result = {"url": "https://example.com", "title": "Example", "content": "Hello"}
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(return_value=mock_result)
        resp = await client.post("/scrape", json={"url": "https://example.com"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com"
    assert body["title"] == "Example"
    assert body["content"] == "Hello"


@pytest.mark.asyncio
async def test_scrape_with_selector(client):
    mock_result = {"url": "https://example.com", "title": "Example", "content": "Selected"}
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(return_value=mock_result)
        resp = await client.post("/scrape", json={"url": "https://example.com", "selector": "main"})

    assert resp.status_code == 200
    mock_svc.scrape.assert_called_once_with(
        url="https://example.com", selector="main", timeout_ms=15000,
    )


# ---------------------------------------------------------------------------
# /scrape — error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_dependency_missing_503(client):
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(side_effect=BrowserDependencyMissingError("no browser"))
        resp = await client.post("/scrape", json={"url": "https://example.com"})

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_scrape_remote_runner_error_502(client):
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(side_effect=BrowserRemoteRunnerError("remote down"))
        resp = await client.post("/scrape", json={"url": "https://example.com"})

    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_scrape_service_error_400(client):
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(side_effect=BrowserServiceError("invalid url"))
        resp = await client.post("/scrape", json={"url": "https://example.com"})

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_scrape_unexpected_error_500(client):
    with patch("life_core.browser_runner_api.browser_service") as mock_svc:
        mock_svc.scrape = AsyncMock(side_effect=RuntimeError("oops"))
        resp = await client.post("/scrape", json={"url": "https://example.com"})

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /scrape — validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_empty_url_rejected(client):
    resp = await client.post("/scrape", json={"url": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_scrape_missing_url_rejected(client):
    resp = await client.post("/scrape", json={})
    assert resp.status_code == 422
