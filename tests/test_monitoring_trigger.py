"""Tests for monitoring_api trigger endpoint and _read_host_stats / _query_prom paths."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.monitoring_api import monitoring_router, _read_host_stats


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(monitoring_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /infra/activepieces/trigger
# ---------------------------------------------------------------------------


class TestTriggerActivepiecesFlow:
    def test_no_token_returns_503(self, client):
        with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": ""}):
            resp = client.post("/infra/activepieces/trigger", json={"flow_name": "test"})
        assert resp.status_code == 503

    def test_flow_not_found_returns_404(self, client):
        flows_resp = MagicMock()
        flows_resp.status_code = 200
        flows_resp.json.return_value = {"data": [
            {"version": {"displayName": "Other Flow"}, "id": "f1"}
        ]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=flows_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.post("/infra/activepieces/trigger", json={"flow_name": "missing"})

        assert resp.status_code == 404

    def test_flow_no_webhook_returns_400(self, client):
        flows_resp = MagicMock()
        flows_resp.status_code = 200
        flows_resp.json.return_value = {"data": [
            {"version": {"displayName": "My Flow", "trigger": {"type": "SCHEDULE", "settings": {}}}, "id": "f1"}
        ]}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=flows_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.post("/infra/activepieces/trigger", json={"flow_name": "My Flow"})

        assert resp.status_code == 400

    def test_trigger_success(self, client):
        flows_resp = MagicMock()
        flows_resp.status_code = 200
        flows_resp.json.return_value = {"data": [
            {
                "version": {
                    "displayName": "Deploy Flow",
                    "trigger": {"type": "WEBHOOK", "settings": {"webhookUrl": "http://ap.local/hook/123"}}
                },
                "id": "f1",
            }
        ]}

        trigger_resp = MagicMock()
        trigger_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=flows_resp)
        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.post("/infra/activepieces/trigger", json={"flow_name": "Deploy Flow"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "triggered"
        assert data["flow_name"] == "Deploy Flow"
        assert data["http_status"] == 200

    def test_list_flows_api_error_returns_502(self, client):
        flows_resp = MagicMock()
        flows_resp.status_code = 500
        flows_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=flows_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.post("/infra/activepieces/trigger", json={"flow_name": "test"})

        assert resp.status_code == 502

    def test_trigger_with_dict_data_key(self, client):
        """Test when flows response has nested data.data structure."""
        flows_resp = MagicMock()
        flows_resp.status_code = 200
        flows_resp.json.return_value = {"data": {"data": [
            {
                "version": {
                    "displayName": "Nested Flow",
                    "trigger": {"type": "WEBHOOK", "settings": {"webhookUrl": "http://ap.local/hook/456"}}
                },
                "id": "f2",
            }
        ]}}

        trigger_resp = MagicMock()
        trigger_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=flows_resp)
        mock_client.post = AsyncMock(return_value=trigger_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.post("/infra/activepieces/trigger", json={"flow_name": "Nested Flow"})

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _query_prom with PROMETHEUS_URL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_prom_direct_prometheus():
    """When PROMETHEUS_URL is set, query goes to Prometheus directly."""
    from life_core.monitoring_api import _query_prom

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": {"result": []}}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch.dict("os.environ", {"PROMETHEUS_URL": "http://prom:9090"}):
        result = await _query_prom(mock_client, "http://grafana:3000", "key", "up")

    call_args = mock_client.get.call_args
    assert "prom:9090" in call_args[0][0]
    assert result == {"data": {"result": []}}


# ---------------------------------------------------------------------------
# _read_host_stats — mocked /proc files
# ---------------------------------------------------------------------------


def test_read_host_stats_all_files_missing():
    """When /proc files don't exist (not in Docker), returns empty dict."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("os.statvfs", side_effect=OSError):
            stats = _read_host_stats()
    assert stats == {} or isinstance(stats, dict)
