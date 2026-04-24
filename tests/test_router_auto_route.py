"""`auto` must resolve to the primary model and succeed through /v1/chat/completions."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_auto_resolves_to_primary_model(monkeypatch):
    from life_core.api import app, resolve_model_alias
    assert resolve_model_alias("auto") == "openai/qwen-14b-awq-kxkm"

    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")

    fake_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/qwen-14b-awq-kxkm",
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
        new=AsyncMock(return_value=fake_response),
    ):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer t"},
            json={"model": "auto", "messages": [
                {"role": "user", "content": "hi"}
            ]},
        )
    assert resp.status_code == 200
