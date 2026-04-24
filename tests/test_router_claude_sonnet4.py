"""V1.8 Axis 11 Task 12 — claude-sonnet-4 dated id must route via /v1/chat/completions."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_claude_sonnet4_dated_id_routes_to_anthropic(monkeypatch):
    """The pinned 2025-05-14 dated id must reach the backend with its
    ``anthropic/`` prefix intact so LiteLLM dispatches to Anthropic."""
    from life_core.api import app

    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")

    captured: dict = {}

    async def fake_call(payload):
        captured["payload"] = payload
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    with patch(
        "life_core.api.call_backend_chat",
        new=AsyncMock(side_effect=fake_call),
    ):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "anthropic/claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    sent_with = captured["payload"]
    assert sent_with["model"].startswith("anthropic/")
    assert sent_with["model"] == "anthropic/claude-sonnet-4-20250514"
