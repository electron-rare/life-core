"""docs/ops/cost-dashboard.md must document thresholds and datasource."""
from __future__ import annotations

from pathlib import Path


def test_cost_dashboard_doc_covers_required_sections():
    p = Path(__file__).resolve().parents[2] / "docs" / "ops" / "cost-dashboard.md"
    text = p.read_text()
    for section in (
        "## Datasource",
        "## Panels",
        "## Alerts",
        "## Runbook",
        "inner_trace.cost_ledger",
        "f4l-cost-overview",
    ):
        assert section in text, f"missing: {section}"
