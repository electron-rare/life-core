"""Extended tests for ws_alerts — exception branches and WebSocket endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from life_core.ws_alerts import _collect_alerts


# ---------------------------------------------------------------------------
# _collect_alerts — exception swallowing for machines, containers, flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_alerts_machines_exception_swallowed():
    """If list_machines raises, other sources still run."""
    gpu_data = {"kv_cache_usage_percent": 30.0}
    containers_data = {"containers": []}
    flows_data = {"flows": [{"name": "flow1", "last_run_status": "FAILED"}]}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, side_effect=RuntimeError("ssh timeout")),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 1
    assert alerts[0]["source"] == "activepieces"


@pytest.mark.asyncio
async def test_collect_alerts_containers_exception_swallowed():
    """If list_containers raises, other sources still run."""
    gpu_data = {"kv_cache_usage_percent": 30.0}
    machines_data = {"machines": [{"name": "tower", "cpu_percent": 95.0}]}
    flows_data = {"flows": []}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, side_effect=RuntimeError("docker api down")),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, return_value=flows_data),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 1
    assert alerts[0]["source"] == "machines"


@pytest.mark.asyncio
async def test_collect_alerts_flows_exception_swallowed():
    """If activepieces_flows raises, other sources still run."""
    gpu_data = {"error": "vllm down"}
    machines_data = {"machines": []}
    containers_data = {"containers": []}

    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, return_value=gpu_data),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, return_value=machines_data),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, return_value=containers_data),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, side_effect=RuntimeError("AP unreachable")),
    ):
        alerts = await _collect_alerts()

    assert len(alerts) == 1
    assert alerts[0]["source"] == "gpu"


@pytest.mark.asyncio
async def test_collect_alerts_all_sources_fail():
    """If all sources raise, we get an empty list — no crash."""
    with (
        patch("life_core.monitoring_api.gpu_stats", new_callable=AsyncMock, side_effect=RuntimeError("err")),
        patch("life_core.monitoring_api.list_machines", new_callable=AsyncMock, side_effect=RuntimeError("err")),
        patch("life_core.infra_api.list_containers", new_callable=AsyncMock, side_effect=RuntimeError("err")),
        patch("life_core.monitoring_api.activepieces_flows", new_callable=AsyncMock, side_effect=RuntimeError("err")),
    ):
        alerts = await _collect_alerts()

    assert alerts == []
