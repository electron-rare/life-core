"""Validate grafana/dashboards/f4l-cost-overview.json."""
from __future__ import annotations

import json
from pathlib import Path


def test_cost_dashboard_shape():
    p = Path(__file__).resolve().parents[2] / "grafana" / "dashboards" / "f4l-cost-overview.json"
    data = json.loads(p.read_text())
    assert data["title"] == "F4L Cost Overview"
    assert data["uid"] == "f4l-cost-overview"
    titles = {p["title"] for p in data["panels"]}
    assert "Daily cost by model" in titles
    assert "Daily cost by user" in titles
    assert "Tokens per day by model" in titles
