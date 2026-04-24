"""`/traces/inner` must return recent inner_trace rows."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_inner_traces_endpoint_returns_rows():
    from life_core.api import app

    fake_rows = [
        {
            "id": "gen-1",
            "agent_run_id": "run-1",
            "llm_model": "openai/qwen-14b-awq-kxkm",
            "tokens_in": 10,
            "tokens_out": 5,
            "cost_usd": 0.0001,
            "status": "success",
            "started_at": "2026-04-24T19:00:00Z",
        }
    ]
    with patch(
        "life_core.traces_api._fetch_inner_traces",
        return_value=fake_rows,
    ):
        client = TestClient(app)
        resp = client.get("/traces/inner?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"][0]["llm_model"] == "openai/qwen-14b-awq-kxkm"
    assert len(body["data"]) == 1
