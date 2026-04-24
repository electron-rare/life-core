"""V1.7 Task 10 — SSE streaming relay tests.

The streaming path must route through ChatService → Router →
LiteLLMProvider so it inherits per-model ``api_base``/``api_key``
resolution. Tests stub ``litellm.acompletion`` at the provider level
(not the shim handler) to exercise the full kwargs chain.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from life_core import api as api_module
from life_core.api import app
from life_core.cache import MultiTierCache
from life_core.router import LiteLLMProvider, Router
from life_core.services import ChatService


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


def _make_chunk(delta_content: str | None, finish_reason: str | None = None):
    """Build a minimal pydantic-free litellm-shaped stream chunk."""
    return SimpleNamespace(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=0,
        model="openai/qwen-14b-awq-kxkm",
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=delta_content, role=None, tool_calls=None
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _install_test_chat_service(monkeypatch) -> ChatService:
    """Wire a real Router + LiteLLMProvider + ChatService into the shim."""
    router = Router()
    provider = LiteLLMProvider(models=["openai/qwen-14b-awq-kxkm"])
    router.register_provider(provider, is_primary=True)
    chat_service = ChatService(router=router, cache=MultiTierCache(redis_url=None))
    monkeypatch.setattr(api_module, "chat_service", chat_service)
    return chat_service


def test_stream_true_returns_event_stream(monkeypatch):
    """Happy path: stream=true emits SSE frames ending with [DONE]."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    _install_test_chat_service(monkeypatch)

    async def fake_acompletion(**kwargs):
        async def gen():
            yield _make_chunk("he")
            yield _make_chunk("llo", finish_reason="stop")

        return gen()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        new=AsyncMock(side_effect=fake_acompletion),
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


def test_stream_forwards_tools_and_tool_choice(monkeypatch):
    """tools / tool_choice / temperature / max_tokens must reach
    ``litellm.acompletion`` even on the streaming path."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    _install_test_chat_service(monkeypatch)

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs

        async def gen():
            yield _make_chunk("ok", finish_reason="stop")

        return gen()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        new=AsyncMock(side_effect=fake_acompletion),
    ):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer t"},
            json={
                "model": "openai/qwen-14b-awq-kxkm",
                "messages": [{"role": "user", "content": "weather paris"}],
                "tools": TOOLS_PAYLOAD,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": 42,
                "stream": True,
            },
        )

    assert resp.status_code == 200
    kwargs = captured["kwargs"]
    assert kwargs.get("stream") is True
    assert kwargs.get("tools") == TOOLS_PAYLOAD
    assert kwargs.get("tool_choice") == "auto"
    assert kwargs.get("temperature") == 0.3
    assert kwargs.get("max_tokens") == 42


def test_stream_frames_parse_as_openai_chunks(monkeypatch):
    """Each non-terminal frame must be a valid OpenAI-compat chunk dict."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    _install_test_chat_service(monkeypatch)

    async def fake_acompletion(**kwargs):
        async def gen():
            yield _make_chunk("he")
            yield _make_chunk("llo", finish_reason="stop")

        return gen()

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        new=AsyncMock(side_effect=fake_acompletion),
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

    body = resp.content.decode()
    frames = [line for line in body.splitlines() if line.startswith("data: ")]
    payloads = [f.removeprefix("data: ") for f in frames]
    assert payloads[-1] == "[DONE]"
    parsed = [json.loads(p) for p in payloads[:-1]]
    assert parsed, "expected at least one JSON chunk before [DONE]"
    for chunk in parsed:
        assert "choices" in chunk
        assert isinstance(chunk["choices"], list)
