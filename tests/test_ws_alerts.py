"""Tests for ws_alerts.py — alert detection logic and async orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from life_core.ws_alerts import (
    _check_containers,
    _check_flows,
    _check_gpu,
    _check_machines,
    _collect_alerts,
)


# ---------------------------------------------------------------------------
# _check_gpu
# ---------------------------------------------------------------------------


def test_check_gpu_healthy():
    gpu = {"kv_cache_usage_percent": 50.0, "running_requests": 2}
    assert _check_gpu(gpu) == []


def test_check_gpu_error():
    gpu = {"error": "connection refused"}
    alerts = _check_gpu(gpu)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "critical"
    assert a["source"] == "gpu"
    assert "GPU inference down" in a["title"]


def test_check_gpu_kv_cache_at_threshold():
    # Exactly 95 — not above, so no alert
    gpu = {"kv_cache_usage_percent": 95.0}
    assert _check_gpu(gpu) == []


def test_check_gpu_kv_cache_above_threshold():
    gpu = {"kv_cache_usage_percent": 97.3}
    alerts = _check_gpu(gpu)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "critical"
    assert a["source"] == "gpu"
    assert "97.3%" in a["message"]
    assert "VRAM critical" in a["title"]


def test_check_gpu_error_takes_precedence_over_kv():
    # error key present — should fire GPU down, not KV cache alert
    gpu = {"error": "timeout", "kv_cache_usage_percent": 99.0}
    alerts = _check_gpu(gpu)
    assert len(alerts) == 1
    assert "GPU inference down" in alerts[0]["title"]


def test_check_gpu_missing_kv_key():
    # no kv_cache_usage_percent key — defaults to 0, no alert
    gpu = {"running_requests": 0}
    assert _check_gpu(gpu) == []


# ---------------------------------------------------------------------------
# _check_containers
# ---------------------------------------------------------------------------


def test_check_containers_all_healthy():
    containers = [
        {"name": "life-core", "health": "healthy"},
        {"name": "life-reborn", "health": "healthy"},
        {"name": "redis", "health": "healthy"},
        {"name": "traefik", "health": "healthy"},
    ]
    assert _check_containers(containers) == []


def test_check_containers_docker_api_error():
    containers = [{"error": "Docker API unreachable"}]
    alerts = _check_containers(containers)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "warning"
    assert a["source"] == "containers"
    assert "unavailable" in a["title"].lower()


def test_check_containers_core_container_unhealthy():
    containers = [
        {"name": "life-core", "health": "unhealthy"},
        {"name": "redis", "health": "healthy"},
    ]
    alerts = _check_containers(containers)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "warning"
    assert "life-core" in a["title"]
    assert a["source"] == "containers"


def test_check_containers_non_core_unhealthy_ignored():
    containers = [
        {"name": "grafana", "health": "unhealthy"},
        {"name": "life-core", "health": "healthy"},
    ]
    assert _check_containers(containers) == []


def test_check_containers_multiple_core_unhealthy():
    containers = [
        {"name": "life-core", "health": "unhealthy"},
        {"name": "redis", "health": "unhealthy"},
    ]
    alerts = _check_containers(containers)
    assert len(alerts) == 2
    sources = {a["source"] for a in alerts}
    assert sources == {"containers"}
    titles = {a["title"] for a in alerts}
    assert "Container degraded: life-core" in titles
    assert "Container degraded: redis" in titles


def test_check_containers_error_stops_after_first():
    # Once an error dict is found, we break — only one alert even with multiple entries
    containers = [
        {"error": "timeout"},
        {"name": "life-core", "health": "unhealthy"},
    ]
    alerts = _check_containers(containers)
    assert len(alerts) == 1
    assert "unavailable" in alerts[0]["title"].lower()


# ---------------------------------------------------------------------------
# _check_machines
# ---------------------------------------------------------------------------


def test_check_machines_normal():
    machines = [
        {"name": "tower", "cpu_percent": 45.0},
        {"name": "cils", "cpu_percent": 70.0},
    ]
    assert _check_machines(machines) == []


def test_check_machines_cpu_above_90():
    machines = [{"name": "tower", "cpu_percent": 95.0}]
    alerts = _check_machines(machines)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "warning"
    assert "tower" in a["title"]
    assert "95" in a["message"]
    assert a["source"] == "machines"


def test_check_machines_exactly_90_no_alert():
    machines = [{"name": "tower", "cpu_percent": 90.0}]
    assert _check_machines(machines) == []


def test_check_machines_multiple_overloaded():
    machines = [
        {"name": "tower", "cpu_percent": 92.0},
        {"name": "kxkm-ai", "cpu_percent": 98.0},
        {"name": "cils", "cpu_percent": 30.0},
    ]
    alerts = _check_machines(machines)
    assert len(alerts) == 2
    names = {a["title"].split(": ")[1] for a in alerts}
    assert names == {"tower", "kxkm-ai"}


def test_check_machines_missing_cpu_key():
    # Missing cpu_percent — defaults to 0, no alert
    machines = [{"name": "mystery"}]
    assert _check_machines(machines) == []


def test_check_machines_empty():
    assert _check_machines([]) == []


# ---------------------------------------------------------------------------
# _check_flows
# ---------------------------------------------------------------------------


def test_check_flows_all_ok():
    flows = [
        {"name": "sync-erp", "last_run_status": "SUCCESS"},
        {"name": "notify", "last_run_status": "SUCCESS"},
    ]
    assert _check_flows(flows) == []


def test_check_flows_one_failed():
    flows = [{"name": "sync-erp", "last_run_status": "FAILED"}]
    alerts = _check_flows(flows)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["severity"] == "warning"
    assert "sync-erp" in a["title"]
    assert "sync-erp" in a["message"]
    assert a["source"] == "activepieces"


def test_check_flows_mixed():
    flows = [
        {"name": "sync-erp", "last_run_status": "SUCCESS"},
        {"name": "notify", "last_run_status": "FAILED"},
        {"name": "backup", "last_run_status": "FAILED"},
    ]
    alerts = _check_flows(flows)
    assert len(alerts) == 2
    failed_names = {a["title"].split(": ")[1] for a in alerts}
    assert failed_names == {"notify", "backup"}


def test_check_flows_no_last_run_status():
    # Missing last_run_status — not "FAILED", no alert
    flows = [{"name": "sync-erp"}]
    assert _check_flows(flows) == []


def test_check_flows_empty():
    assert _check_flows([]) == []


# ---------------------------------------------------------------------------
# _collect_alerts (async orchestrator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_alerts_all_healthy():
    gpu_data = {"kv_cache_usage_percent": 30.0}
    machines_data = {"machines": [{"name": "tower", "cpu_percent": 50.0}]}
    containers_data = {"containers": [{"name": "life-core", "health": "healthy"}]}
    flows_data = {"flows": [{"name": "sync-erp", "last_run_status": "SUCCESS"}]}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert alerts == []


@pytest.mark.asyncio
async def test_collect_alerts_gpu_critical():
    gpu_data = {"error": "vLLM down"}
    machines_data = {"machines": []}
    containers_data = {"containers": []}
    flows_data = {"flows": []}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"
    assert "id" in alerts[0]
    assert "timestamp" in alerts[0]


@pytest.mark.asyncio
async def test_collect_alerts_attaches_id_and_timestamp():
    gpu_data = {"kv_cache_usage_percent": 99.0}
    machines_data = {"machines": []}
    containers_data = {"containers": []}
    flows_data = {"flows": []}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 1
    a = alerts[0]
    assert a["id"].startswith("alert-")
    # Timestamp is ISO-like: YYYY-MM-DDTHH:MM:SSZ
    assert "T" in a["timestamp"] and a["timestamp"].endswith("Z")


@pytest.mark.asyncio
async def test_collect_alerts_multiple_sources():
    gpu_data = {"error": "timeout"}
    machines_data = {"machines": [{"name": "tower", "cpu_percent": 95.0}]}
    containers_data = {"containers": [{"name": "life-core", "health": "unhealthy"}]}
    flows_data = {"flows": [{"name": "sync-erp", "last_run_status": "FAILED"}]}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 4
    sources = {a["source"] for a in alerts}
    assert sources == {"gpu", "machines", "containers", "activepieces"}
    # Each alert has unique id
    ids = [a["id"] for a in alerts]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_collect_alerts_endpoint_exception_is_swallowed():
    """If one endpoint raises, others still run."""
    machines_data = {"machines": []}
    containers_data = {"containers": []}
    flows_data = {"flows": [{"name": "fail-flow", "last_run_status": "FAILED"}]}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, side_effect=RuntimeError("net error")),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    # GPU check failed silently; flows check produced one alert
    assert len(alerts) == 1
    assert alerts[0]["source"] == "activepieces"
