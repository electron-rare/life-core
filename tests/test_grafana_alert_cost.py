"""Validate grafana/alerts/cost-anomaly.yaml shape."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_alert_rule_shape():
    p = Path(__file__).resolve().parents[2] / "grafana" / "alerts" / "cost-anomaly.yaml"
    data = yaml.safe_load(p.read_text())
    groups = {g["name"] for g in data["groups"]}
    assert "f4l-cost" in groups
    rules = data["groups"][0]["rules"]
    names = {r["alert"] for r in rules}
    assert "F4LDailyCostAnomaly" in names
