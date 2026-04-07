"""Tests for monitoring_api: machines, GPU stats, Activepieces flows."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.monitoring_api import (
    monitoring_router,
    _extract_by_instance,
    _parse_prometheus_text,
    _query_prom,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(monitoring_router)
    return TestClient(app)


def _prom_vector(instance: str, value: str) -> dict:
    return {
        "data": {
            "result": [
                {"metric": {"instance": instance}, "value": [1700000000, value]}
            ]
        }
    }


def _prom_vector_multi(entries: list[tuple[str, str]]) -> dict:
    return {
        "data": {
            "result": [
                {"metric": {"instance": inst}, "value": [1700000000, val]}
                for inst, val in entries
            ]
        }
    }


# ---------------------------------------------------------------------------
# Pure function: _extract_by_instance
# ---------------------------------------------------------------------------


class TestExtractByInstance:
    def test_single_entry(self):
        data = _prom_vector("192.168.0.120:9100", "55.3")
        result = _extract_by_instance(data)
        assert result == {"192.168.0.120:9100": 55.3}

    def test_multiple_entries(self):
        data = _prom_vector_multi([
            ("192.168.0.120:9100", "10.0"),
            ("192.168.0.119:9100", "20.5"),
        ])
        result = _extract_by_instance(data)
        assert result["192.168.0.120:9100"] == 10.0
        assert result["192.168.0.119:9100"] == 20.5

    def test_empty_result(self):
        data = {"data": {"result": []}}
        assert _extract_by_instance(data) == {}

    def test_missing_data_key(self):
        assert _extract_by_instance({}) == {}

    def test_bad_value_skipped(self):
        data = {
            "data": {
                "result": [
                    {"metric": {"instance": "host:9100"}, "value": [0, "NaN_BAD"]},
                    {"metric": {"instance": "host2:9100"}, "value": [0, "42.0"]},
                ]
            }
        }
        result = _extract_by_instance(data)
        assert "host:9100" not in result
        assert result["host2:9100"] == 42.0

    def test_missing_value_key_skipped(self):
        data = {"data": {"result": [{"metric": {"instance": "host:9100"}}]}}
        assert _extract_by_instance(data) == {}

    def test_missing_instance_label(self):
        data = {
            "data": {
                "result": [{"metric": {}, "value": [0, "1.0"]}]
            }
        }
        result = _extract_by_instance(data)
        assert result.get("") == 1.0


# ---------------------------------------------------------------------------
# Pure function: _parse_prometheus_text
# ---------------------------------------------------------------------------


PROM_TEXT_SAMPLE = """\
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage percent
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"} 0.42
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"} 3
# HELP vllm:generation_tokens_total Total generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"} 12345.0
"""


class TestParsePrometheusText:
    def test_parses_gauge_value(self):
        metrics = _parse_prometheus_text(PROM_TEXT_SAMPLE)
        assert metrics.get("vllm:gpu_cache_usage_perc") == pytest.approx(0.42)

    def test_parses_integer_gauge(self):
        metrics = _parse_prometheus_text(PROM_TEXT_SAMPLE)
        assert metrics.get("vllm:num_requests_running") == pytest.approx(3.0)

    def test_parses_counter(self):
        metrics = _parse_prometheus_text(PROM_TEXT_SAMPLE)
        assert metrics.get("vllm:generation_tokens_total") == pytest.approx(12345.0)

    def test_ignores_comments(self):
        text = "# HELP foo bar\n# TYPE foo gauge\nfoo 1.0\n"
        metrics = _parse_prometheus_text(text)
        assert "foo" in metrics
        assert metrics["foo"] == 1.0

    def test_ignores_blank_lines(self):
        text = "\nfoo 1.0\n\nbar 2.0\n"
        metrics = _parse_prometheus_text(text)
        assert metrics["foo"] == 1.0
        assert metrics["bar"] == 2.0

    def test_strips_labels_for_lookup(self):
        text = 'some_metric{label="val"} 9.9\n'
        metrics = _parse_prometheus_text(text)
        assert metrics.get("some_metric") == pytest.approx(9.9)

    def test_empty_text(self):
        assert _parse_prometheus_text("") == {}

    def test_bad_line_ignored(self):
        text = "not_a_valid_line\nfoo 1.0\n"
        metrics = _parse_prometheus_text(text)
        assert "foo" in metrics


# ---------------------------------------------------------------------------
# Async: _query_prom
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_prom_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": {"result": []}}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    result = await _query_prom(mock_client, "http://grafana:3000", "mykey", "up")

    mock_client.get.assert_awaited_once()
    call_kwargs = mock_client.get.call_args
    assert "Authorization" in call_kwargs.kwargs["headers"]
    assert result == {"data": {"result": []}}


@pytest.mark.asyncio
async def test_query_prom_no_api_key():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    await _query_prom(mock_client, "http://grafana:3000", "", "up")

    call_kwargs = mock_client.get.call_args
    assert call_kwargs.kwargs["headers"] == {}


# ---------------------------------------------------------------------------
# Endpoint: /infra/machines
# ---------------------------------------------------------------------------


def _make_prom_response(value: str = "0") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": {"result": []}}
    return mock_resp


class TestListMachines:
    def test_returns_four_machines_on_success(self, client):
        empty_prom = {"data": {"result": []}}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = empty_prom

        async def fake_get(*args, **kwargs):
            return mock_resp

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(side_effect=fake_get)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"GRAFANA_URL": "http://grafana:3000", "GRAFANA_API_KEY": "tok"}):
                resp = client.get("/infra/machines")

        assert resp.status_code == 200
        data = resp.json()
        assert "machines" in data
        assert len(data["machines"]) == 4

    def test_machine_names(self, client):
        empty_prom = {"data": {"result": []}}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = empty_prom

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            resp = client.get("/infra/machines")

        names = {m["name"] for m in resp.json()["machines"]}
        assert names == {"Tower", "KXKM-AI", "Cils", "GrosMac"}

    def test_fallback_on_proc_unavailable(self, client):
        """Remote machines get 'remote_no_agent' error, Tower reads /proc (may fallback)."""
        with patch("life_core.monitoring_api._read_host_stats", return_value={}):
            resp = client.get("/infra/machines")

        assert resp.status_code == 200
        machines = resp.json()["machines"]
        assert len(machines) == 4
        remote = [m for m in machines if m["name"] != "Tower"]
        assert all(m.get("error") == "remote_no_agent" for m in remote)

    def test_fallback_machine_fields(self, client):
        """Tower uses /proc stats; with empty stats it falls back to MACHINE_DEFAULTS."""
        with patch("life_core.monitoring_api._read_host_stats", return_value={}):
            resp = client.get("/infra/machines")

        tower = next(m for m in resp.json()["machines"] if m["name"] == "Tower")
        assert tower["ram_total_gb"] == pytest.approx(31.0, abs=1.0)
        assert tower["disk_total_gb"] == pytest.approx(1800.0, abs=100.0)
        assert tower["cpu_percent"] == 0.0

    def test_with_real_proc_data(self, client):
        """Tower reads /proc stats; verify cpu, ram and uptime are populated."""
        fake_stats = {
            "cpu_idle_ratio": 0.575,          # → cpu_percent = 42.5
            "ram_total": 31 * 1024**3,
            "ram_available": 8 * 1024**3,
            "disk_total": 1800 * 1024**3,
            "disk_used": 100 * 1024**3,
            "uptime_seconds": 3600 * 5,       # 5 hours
        }

        with patch("life_core.monitoring_api._read_host_stats", return_value=fake_stats):
            resp = client.get("/infra/machines")

        tower = next(m for m in resp.json()["machines"] if m["name"] == "Tower")
        assert tower["cpu_percent"] == pytest.approx(42.5, abs=0.1)
        assert tower["ram_total_gb"] == pytest.approx(31.0, abs=0.1)
        assert tower["uptime_hours"] == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Endpoint: /infra/gpu
# ---------------------------------------------------------------------------


class TestGpuStats:
    def test_no_vllm_url_returns_fallback(self, client):
        env = {"VLLM_METRICS_URL": "", "VLLM_BASE_URL": ""}
        with patch.dict("os.environ", env, clear=False):
            resp = client.get("/infra/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "vllm_unreachable"
        assert data["vram_total_gb"] == 24.0

    def test_success_path(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = PROM_TEXT_SAMPLE

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"VLLM_METRICS_URL": "http://kxkm:11434/metrics"}):
                resp = client.get("/infra/gpu")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
        assert data["requests_active"] == 3
        assert data["kv_cache_usage_percent"] == pytest.approx(42.0, abs=0.1)
        assert data["vram_total_gb"] == 24.0
        assert "error" not in data

    def test_vllm_unreachable_returns_fallback(self, client):
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"VLLM_METRICS_URL": "http://kxkm:11434/metrics"}):
                resp = client.get("/infra/gpu")

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "vllm_unreachable"

    def test_vllm_base_url_fallback(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = PROM_TEXT_SAMPLE

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"VLLM_METRICS_URL": "", "VLLM_BASE_URL": "http://kxkm:11434"}):
                resp = client.get("/infra/gpu")

        assert resp.status_code == 200
        called_url = mock_async_client.get.call_args[0][0]
        assert called_url.endswith("/metrics")


# ---------------------------------------------------------------------------
# Endpoint: /infra/activepieces
# ---------------------------------------------------------------------------


class TestActivepiecesFlows:
    def test_no_token_returns_note(self, client):
        with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": ""}, clear=False):
            resp = client.get("/infra/activepieces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["flows"] == []
        assert "note" in data

    def test_success_path(self, client):
        ap_data = {
            "data": [
                {
                    "id": "flow-1",
                    "status": "ENABLED",
                    "version": {"displayName": "My Flow", "trigger": {"type": "WEBHOOK"}},
                    "lastRun": {"startTime": "2026-04-01T10:00:00Z", "status": "SUCCEEDED"},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ap_data

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "secret-token"}):
                resp = client.get("/infra/activepieces")

        assert resp.status_code == 200
        flows = resp.json()["flows"]
        assert len(flows) == 1
        assert flows[0]["id"] == "flow-1"
        assert flows[0]["name"] == "My Flow"
        assert flows[0]["status"] == "ENABLED"
        assert flows[0]["trigger"] == "WEBHOOK"
        assert flows[0]["last_run_status"] == "SUCCEEDED"

    def test_api_error_returns_fallback(self, client):
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(side_effect=Exception("timeout"))

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "secret-token"}):
                resp = client.get("/infra/activepieces")

        assert resp.status_code == 200
        data = resp.json()
        assert data["flows"] == []
        assert "error" in data

    def test_dict_with_data_key_response_format(self, client):
        """Test when API returns {data: [...]} envelope (standard AP format)."""
        ap_data = {
            "data": [
                {
                    "id": "flow-2",
                    "status": "DISABLED",
                    "version": {"displayName": "Other Flow", "trigger": {"type": "SCHEDULE"}},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ap_data

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "tok"}):
                resp = client.get("/infra/activepieces")

        flows = resp.json()["flows"]
        assert len(flows) == 1
        assert flows[0]["id"] == "flow-2"

    def test_auth_header_sent(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        with patch("life_core.monitoring_api.httpx.AsyncClient", return_value=mock_async_client):
            with patch.dict("os.environ", {"ACTIVEPIECES_TOKEN": "my-secret"}):
                client.get("/infra/activepieces")

        call_kwargs = mock_async_client.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-secret"
