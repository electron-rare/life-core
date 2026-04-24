"""Validate grafana/dashboards/inner-trace-latency.json has the required panels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "grafana"
    / "dashboards"
    / "inner-trace-latency.json"
)


@pytest.mark.skipif(
    not _DASHBOARD_PATH.exists(),
    reason="dashboard file in superrepo not reachable from life-core CI",
)
def test_dashboard_has_latency_panels():
    data = json.loads(_DASHBOARD_PATH.read_text())
    assert data["title"] == "F4L Inner Trace Latency"
    titles = {p["title"] for p in data["panels"]}
    assert "Latency p50 by model" in titles
    assert "Latency p95 by model" in titles
    assert "Latency p99 by model" in titles
