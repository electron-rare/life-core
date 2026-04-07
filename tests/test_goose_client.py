"""Tests for GooseClient ACP protocol client."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.goose_client import GooseClient, GooseSession, GooseMessage


def test_client_init_defaults():
    client = GooseClient()
    assert client.base_url == "http://goosed:3000"


def test_client_init_custom_url():
    client = GooseClient(base_url="http://localhost:9999")
    assert client.base_url == "http://localhost:9999"


def test_session_dataclass():
    s = GooseSession(session_id="abc-123", working_dir="/tmp")
    assert s.session_id == "abc-123"
    assert s.working_dir == "/tmp"


def test_message_dataclass():
    m = GooseMessage(role="assistant", content="hello", tool_calls=[])
    assert m.role == "assistant"
    assert m.content == "hello"


@pytest.mark.asyncio
async def test_create_session():
    client = GooseClient(base_url="http://fake:3000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"acp-session-id": "session-xyz"}
    mock_resp.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

    with patch.object(client, "_rpc", new_callable=AsyncMock, return_value=("session-xyz", {"jsonrpc": "2.0", "result": {}, "id": 1})):
        session = await client.create_session(working_dir="/tmp/test")
    assert session.session_id == "session-xyz"
    assert session.working_dir == "/tmp/test"


@pytest.mark.asyncio
async def test_prompt_collects_text():
    client = GooseClient(base_url="http://fake:3000")

    async def fake_stream(method, params, session_id):
        yield {"jsonrpc": "2.0", "method": "AgentMessageChunk", "params": {"content": "Hello "}}
        yield {"jsonrpc": "2.0", "method": "AgentMessageChunk", "params": {"content": "world"}}

    with patch.object(client, "_stream_rpc", side_effect=fake_stream):
        chunks = []
        async for chunk in client.prompt("session-abc", "say hello"):
            chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[0]["params"]["content"] == "Hello "
    assert chunks[1]["params"]["content"] == "world"


@pytest.mark.asyncio
async def test_prompt_non_streaming():
    client = GooseClient(base_url="http://fake:3000")

    async def fake_stream(method, params, session_id):
        yield {"jsonrpc": "2.0", "method": "AgentMessageChunk", "params": {"content": "done"}}

    with patch.object(client, "_stream_rpc", side_effect=fake_stream):
        result = await client.prompt_sync("session-abc", "do something")
    assert result == "done"


@pytest.mark.asyncio
async def test_cancel_session():
    client = GooseClient(base_url="http://fake:3000")
    with patch.object(client, "_rpc", new_callable=AsyncMock, return_value=(None, {"jsonrpc": "2.0", "result": {}, "id": 1})):
        await client.cancel("session-abc")


@pytest.mark.asyncio
async def test_health_check():
    client = GooseClient(base_url="http://fake:3000")
    with patch("life_core.goose_client.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"status": "ok"}
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client
        result = await client.health()
    assert result["status"] == "ok"
