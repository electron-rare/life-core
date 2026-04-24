"""V1.7 Task 10 — SSE streaming relay tests."""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from life_core.api import app


async def fake_stream_chunks(payload):  # noqa: ARG001
    yield b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
    yield b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
    yield b"data: [DONE]\n\n"


def test_stream_true_returns_event_stream(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    with patch(
        "life_core.api.stream_backend_chunks",
        new=fake_stream_chunks,
    ):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "openai/qwen-14b-awq-kxkm",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.content.decode()
    assert "data: [DONE]" in body
    assert "he" in body and "llo" in body
