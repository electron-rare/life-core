"""V1.8 Axis 11 Task 14 — qwen-14b-awq-kxkm must route via /v1/chat/completions."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_qwen14b_awq_kxkm_routes_to_openai_compat(monkeypatch):
    """The canonical ``openai/qwen-14b-awq-kxkm`` id must reach the
    backend verbatim — LiteLLM dispatches it to the vLLM pool via the
    OpenAI-compat protocol."""
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
                "model": "openai/qwen-14b-awq-kxkm",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    sent_with = captured["payload"]
    assert sent_with["model"] == "openai/qwen-14b-awq-kxkm"
