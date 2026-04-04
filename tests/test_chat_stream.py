"""Tests for /chat/stream SSE endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from life_core.api import app


@pytest.mark.asyncio
async def test_chat_stream_returns_sse():
    """POST /chat/stream should return text/event-stream with delta chunks."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="hello"))]

    async def fake_acompletion(**kwargs):
        async def gen():
            yield mock_chunk

        return gen()

    with patch("life_core.api.chat_service") as mock_service, patch(
        "litellm.acompletion", side_effect=fake_acompletion
    ):
        mock_service.__bool__ = lambda self: True
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/chat/stream",
                json={"messages": [{"role": "user", "content": "hi"}]},
            ) as r:
                assert r.status_code == 200
                assert "text/event-stream" in r.headers["content-type"]
                body = await r.aread()
                assert b"data:" in body


@pytest.mark.asyncio
async def test_chat_stream_done_sentinel():
    """Stream should end with [DONE] sentinel."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="world"))]

    async def fake_acompletion(**kwargs):
        async def gen():
            yield mock_chunk

        return gen()

    with patch("life_core.api.chat_service") as mock_service, patch(
        "litellm.acompletion", side_effect=fake_acompletion
    ):
        mock_service.__bool__ = lambda self: True
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/chat/stream",
                json={"messages": [{"role": "user", "content": "hi"}]},
            ) as r:
                body = await r.aread()
                assert b"[DONE]" in body
