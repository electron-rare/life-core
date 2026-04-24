"""Validate grafana/dashboards/inner-trace-latency.json has the required panels."""
from __future__ import annotations

import json
from pathlib import Path


def test_dashboard_has_latency_panels():
    path = Path(__file__).resolve().parents[2] / "grafana" / "dashboards" / "inner-trace-latency.json"
    data = json.loads(path.read_text())
    assert data["title"] == "F4L Inner Trace Latency"
    titles = {p["title"] for p in data["panels"]}
    assert "Latency p50 by model" in titles
    assert "Latency p95 by model" in titles
    assert "Latency p99 by model" in titles
