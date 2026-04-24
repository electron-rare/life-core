"""ChatService must emit inner_trace rows when completing a request."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_chat_service_records_generation_run(monkeypatch):
    from life_core.services.chat import ChatService

    emitter = MagicMock()
    emitter.record_agent_run.return_value = (
        "00000000-0000-0000-0000-000000000001"
    )
    emitter.record_generation_run.return_value = (
        "00000000-0000-0000-0000-000000000002"
    )

    router = MagicMock()
    router.send = AsyncMock(return_value=MagicMock(
        content="ok",
        usage={"prompt_tokens": 12, "completion_tokens": 5},
        model="openai/qwen-14b-awq-kxkm",
        provider="kxkm",
        cost_usd=0.0001,
    ))

    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)

    svc = ChatService(router=router, cache=cache, trace_emitter=emitter)
    out = await svc.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/qwen-14b-awq-kxkm",
        deliverable_slug="test-chat",
    )

    assert out is not None
    emitter.record_agent_run.assert_called_once()
    emitter.record_generation_run.assert_called_once()
