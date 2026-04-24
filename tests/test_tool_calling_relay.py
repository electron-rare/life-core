"""V1.7 Task 9 — tests for tool-calling relay in the shim."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from life_core.api import app


TOOLS_PAYLOAD = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_tools_field_is_forwarded_to_backend(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    captured = {}

    async def fake_call(payload):
        captured["payload"] = payload
        return {
            "id": "x",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }

    with patch("life_core.api.call_backend_chat", new=AsyncMock(side_effect=fake_call)):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "openai/qwen-14b-awq-kxkm",
                "messages": [{"role": "user", "content": "weather paris"}],
                "tools": TOOLS_PAYLOAD,
                "tool_choice": "auto",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert captured["payload"]["tools"] == TOOLS_PAYLOAD
    assert captured["payload"]["tool_choice"] == "auto"


def test_absence_of_tools_does_not_inject_field(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    captured = {}

    async def fake_call(payload):
        captured["payload"] = payload
        return {
            "id": "y",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
        }

    with patch("life_core.api.call_backend_chat", new=AsyncMock(side_effect=fake_call)):
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
    assert "tools" not in captured["payload"]
    assert "tool_choice" not in captured["payload"]
