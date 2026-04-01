"""Tests for Langfuse prompt versioning in ChatService."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.router.providers.base import LLMResponse
from life_core.services.chat import ChatService


@pytest.mark.asyncio
async def test_chat_uses_langfuse_prompt_when_available():
    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Hi", model="openai/gpt-4o", provider="litellm",
        usage={"input_tokens": 10, "output_tokens": 5},
    ))

    service = ChatService(router=mock_router, cache=None, rag=None)

    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = "You are a helpful manufacturing assistant."

    with patch("life_core.langfuse_tracing.get_langfuse_prompt", return_value=mock_prompt):
        await service.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="openai/gpt-4o",
        )

    call_args = mock_router.send.call_args
    messages_sent = call_args[1].get("messages", call_args[0][0] if call_args[0] else [])
    assert any(m.get("role") == "system" for m in messages_sent)


@pytest.mark.asyncio
async def test_chat_falls_back_without_langfuse():
    mock_router = MagicMock()
    mock_router.send = AsyncMock(return_value=LLMResponse(
        content="Hi", model="openai/gpt-4o", provider="litellm",
        usage={"input_tokens": 5, "output_tokens": 3},
    ))

    service = ChatService(router=mock_router, cache=None, rag=None)

    with patch("life_core.langfuse_tracing.get_langfuse_prompt", return_value=None):
        result = await service.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="openai/gpt-4o",
        )

    assert result["content"] == "Hi"
