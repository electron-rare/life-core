"""V1.7 Task 9 — tests for tool-calling relay in the shim."""
from __future__ import annotations

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


def _fake_litellm_response_with_tool_call() -> SimpleNamespace:
    """Shape matches what ``LiteLLMProvider._to_llm_response`` consumes."""
    return SimpleNamespace(
        model="openai/qwen-14b-awq-kxkm",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="get_weather",
                                arguments='{"city": "Paris"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_tools_flow_through_without_mocking_call_backend(monkeypatch):
    """Integration: tools must reach ``litellm.acompletion``.

    Mocks one level deeper than the shim handler (the LiteLLM SDK
    call itself) to prove chat_service → Router → LiteLLMProvider do
    not drop tools/tool_choice along the way.
    """
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")

    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs
        return _fake_litellm_response_with_tool_call()

    # Build a ChatService with a real Router + LiteLLMProvider so the
    # kwargs chain is exercised end-to-end; only litellm.acompletion
    # is stubbed out.
    router = Router()
    provider = LiteLLMProvider(models=["openai/qwen-14b-awq-kxkm"])
    router.register_provider(provider, is_primary=True)
    chat_service = ChatService(router=router, cache=MultiTierCache(redis_url=None))

    # Inject our chat_service into the module so the shim picks it up
    # instead of the lifespan-initialised global (TestClient context
    # manager would re-run startup otherwise).
    monkeypatch.setattr(api_module, "chat_service", chat_service)

    with patch(
        "life_core.router.providers.litellm_provider.litellm.acompletion",
        new=AsyncMock(side_effect=fake_acompletion),
    ):
        # Avoid litellm.completion_cost noise on the fake response
        with patch(
            "life_core.router.providers.litellm_provider.litellm.completion_cost",
            return_value=0.0,
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
                },
            )

    assert resp.status_code == 200, resp.text
    assert captured["kwargs"].get("tools") == TOOLS_PAYLOAD
    assert captured["kwargs"].get("tool_choice") == "auto"
    assert captured["kwargs"].get("temperature") == 0.3
    assert captured["kwargs"].get("max_tokens") == 42
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    tc = body["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
