"""Tests for GooseClient ACP subprocess client."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from life_core.goose_client import GooseClient, GooseSession


def test_client_init_defaults():
    client = GooseClient()
    assert client.goose_bin == "goose"


def test_client_init_custom_bin():
    client = GooseClient(goose_bin="/usr/local/bin/goose")
    assert client.goose_bin == "/usr/local/bin/goose"


def test_session_dataclass():
    s = GooseSession(session_id="abc-123", working_dir="/tmp")
    assert s.session_id == "abc-123"
    assert s.working_dir == "/tmp"


@pytest.mark.asyncio
async def test_health_binary_found():
    client = GooseClient()
    with patch("life_core.goose_client.shutil.which", return_value="/usr/local/bin/goose"):
        result = await client.health()
    assert result["status"] == "ok"
    assert result["binary"] == "/usr/local/bin/goose"


@pytest.mark.asyncio
async def test_health_binary_not_found():
    client = GooseClient()
    with patch("life_core.goose_client.shutil.which", return_value=None):
        result = await client.health()
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_create_session():
    client = GooseClient()

    async def fake_rpc(method, params=None):
        if method == "initialize":
            return {}
        if method == "session/new":
            return {"session_id": "sess-xyz"}
        return {}

    with patch.object(client, "_rpc", side_effect=fake_rpc):
        with patch.object(client, "_ensure_process", new_callable=AsyncMock):
            session = await client.create_session(working_dir="workspace")
    assert session.session_id == "sess-xyz"
    assert session.working_dir == "workspace"


@pytest.mark.asyncio
async def test_prompt_collects_text():
    client = GooseClient()

    async def fake_stream(method, params=None):
        yield {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "session_id": "s1",
                "update": {
                    "kind": "message",
                    "content": [{"type": "text", "text": "Hello world"}],
                },
            },
        }

    with patch.object(client, "_rpc_stream", side_effect=fake_stream):
        chunks = []
        async for chunk in client.prompt("s1", "say hello"):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0]["method"] == "AgentMessageChunk"
    assert chunks[0]["params"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_prompt_sync():
    client = GooseClient()

    async def fake_stream(method, params=None):
        yield {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "session_id": "s1",
                "update": {
                    "kind": "message",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        }

    with patch.object(client, "_rpc_stream", side_effect=fake_stream):
        result = await client.prompt_sync("s1", "do it")
    assert result == "done"


@pytest.mark.asyncio
async def test_prompt_tool_call():
    client = GooseClient()

    async def fake_stream(method, params=None):
        yield {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "session_id": "s1",
                "update": {
                    "kind": "tool_use",
                    "name": "finefab_rag_search",
                    "input": {"query": "test"},
                },
            },
        }

    with patch.object(client, "_rpc_stream", side_effect=fake_stream):
        chunks = []
        async for chunk in client.prompt("s1", "search"):
            chunks.append(chunk)

    assert chunks[0]["method"] == "ToolCall"
    assert chunks[0]["params"]["name"] == "finefab_rag_search"


@pytest.mark.asyncio
async def test_cancel():
    client = GooseClient()
    with patch.object(client, "_rpc", new_callable=AsyncMock, return_value={}):
        await client.cancel("s1")
