"""User id from Keycloak JWT must land in generation_run row + Langfuse trace."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_chat_service_passes_user_id_to_emitter():
    from life_core.services.chat import ChatService

    emitter = MagicMock()
    emitter.record_agent_run.return_value = "run-1"
    emitter.record_generation_run.return_value = "gen-1"

    response = MagicMock()
    response.content = "ok"
    response.usage = {"prompt_tokens": 1, "completion_tokens": 1}
    response.model = "openai/qwen-14b-awq-kxkm"
    response.provider = "vllm"
    response.cost_usd = 0.0001
    response.tool_calls = None

    router = MagicMock()
    router.send = AsyncMock(return_value=response)

    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)

    svc = ChatService(router=router, cache=cache, trace_emitter=emitter)
    await svc.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/qwen-14b-awq-kxkm",
        deliverable_slug="x",
        user_id="kc-user-42",
    )

    kwargs = emitter.record_generation_run.call_args.kwargs
    assert kwargs["user_id"] == "kc-user-42"
