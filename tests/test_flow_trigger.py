"""Tests for Activepieces flow trigger endpoint."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.monitoring_api import monitoring_router

_app = FastAPI()
_app.include_router(monitoring_router)
client = TestClient(_app)


def test_trigger_flow_missing_token():
    """Should return 503 when token not configured."""
    with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": ""}, clear=False):
        resp = client.post("/infra/activepieces/trigger", json={"flow_name": "test"})
    assert resp.status_code == 503


def test_trigger_flow_api_error():
    """Should return 502 when Activepieces API fails."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock(status_code=500))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "test-token"}, clear=False), \
         patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.post("/infra/activepieces/trigger", json={"flow_name": "test"})
    assert resp.status_code == 502


def test_trigger_flow_not_found():
    """Should return 404 for unknown flow."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock(
        status_code=200,
        json=MagicMock(return_value={"data": []})
    ))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "test-token"}, clear=False), \
         patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.post("/infra/activepieces/trigger", json={"flow_name": "nonexistent"})
    assert resp.status_code == 404


def test_trigger_flow_no_webhook():
    """Should return 400 when flow has no webhook trigger."""
    flow_data = [{"version": {"displayName": "my-flow", "trigger": {"settings": {}}}}]
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock(
        status_code=200,
        json=MagicMock(return_value={"data": flow_data})
    ))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "test-token"}, clear=False), \
         patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.post("/infra/activepieces/trigger", json={"flow_name": "my-flow"})
    assert resp.status_code == 400


def test_trigger_flow_success():
    """Should trigger flow and return success."""
    flow_data = [{
        "version": {
            "displayName": "deploy",
            "trigger": {"settings": {"webhookUrl": "https://auto.saillant.cc/hook/abc"}},
        }
    }]
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock(
        status_code=200,
        json=MagicMock(return_value={"data": flow_data})
    ))
    mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "test-token"}, clear=False), \
         patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.post("/infra/activepieces/trigger", json={"flow_name": "deploy"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "triggered"
    assert data["flow_name"] == "deploy"
    assert data["http_status"] == 200
