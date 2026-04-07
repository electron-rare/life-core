"""Tests for Goose session registry (Redis-backed)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_redis():
    """Return a mock Redis client."""
    r = AsyncMock()
    # hgetall returns {} by default (empty = not found)
    r.hgetall = AsyncMock(return_value={})
    r.hset = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=1)
    r.hincrby = AsyncMock(return_value=1)
    r.delete = AsyncMock(return_value=1)
    r.scan = AsyncMock(return_value=(0, []))
    return r


@pytest.fixture
def registry(mock_redis):
    """Return a SessionRegistry with a mocked Redis client."""
    from life_core.goose_sessions import SessionRegistry
    reg = SessionRegistry.__new__(SessionRegistry)
    reg._redis = mock_redis
    return reg


@pytest.mark.asyncio
async def test_register_returns_session_info(registry, mock_redis):
    """register() should store a hash and return a SessionInfo."""
    from life_core.goose_sessions import SessionInfo
    info = await registry.register("sess-1", "workspace/proj")
    assert isinstance(info, SessionInfo)
    assert info.session_id == "sess-1"
    assert info.working_dir == "workspace/proj"
    assert info.message_count == 0
    # Should have called hset and expire
    mock_redis.hset.assert_called_once()
    mock_redis.expire.assert_called_once()


@pytest.mark.asyncio
async def test_register_key_format(registry, mock_redis):
    """register() should store at goose:session:{id}."""
    await registry.register("abc123", ".")
    call_args = mock_redis.hset.call_args
    assert call_args[0][0] == "goose:session:abc123"


@pytest.mark.asyncio
async def test_register_ttl_24h(registry, mock_redis):
    """register() should set a 24-hour TTL."""
    await registry.register("sess-2", ".")
    expire_args = mock_redis.expire.call_args
    assert expire_args[0][0] == "goose:session:sess-2"
    assert expire_args[0][1] == 86400  # 24 * 3600


@pytest.mark.asyncio
async def test_touch_updates_last_active_and_count(registry, mock_redis):
    """touch() should update last_active, increment message_count, reset TTL."""
    mock_redis.hgetall.return_value = {
        b"session_id": b"s1",
        b"working_dir": b".",
        b"created_at": b"2026-04-07T10:00:00",
        b"last_active": b"2026-04-07T10:00:00",
        b"message_count": b"3",
    }
    await registry.touch("s1")
    # hset called to update last_active
    mock_redis.hset.assert_called_once()
    # hincrby called to increment message_count
    mock_redis.hincrby.assert_called_once_with("goose:session:s1", "message_count", 1)
    # TTL reset
    mock_redis.expire.assert_called_once_with("goose:session:s1", 86400)


@pytest.mark.asyncio
async def test_touch_nonexistent_is_noop(registry, mock_redis):
    """touch() on a missing session should not crash."""
    mock_redis.hgetall.return_value = {}
    await registry.touch("missing")
    mock_redis.hset.assert_not_called()


@pytest.mark.asyncio
async def test_list_sessions_returns_sorted(registry, mock_redis):
    """list_sessions() should return sessions sorted by last_active desc."""
    keys = [b"goose:session:a", b"goose:session:b"]

    async def fake_scan_iter(**kwargs):
        for k in keys:
            yield k
    mock_redis.scan_iter = fake_scan_iter

    def hgetall_side(key):
        data = {
            b"goose:session:a": {
                b"session_id": b"a",
                b"working_dir": b".",
                b"created_at": b"2026-04-07T09:00:00",
                b"last_active": b"2026-04-07T09:00:00",
                b"message_count": b"2",
            },
            b"goose:session:b": {
                b"session_id": b"b",
                b"working_dir": b"proj",
                b"created_at": b"2026-04-07T08:00:00",
                b"last_active": b"2026-04-07T11:00:00",
                b"message_count": b"5",
            },
        }
        return data.get(key, {})

    mock_redis.hgetall = AsyncMock(side_effect=hgetall_side)
    sessions = await registry.list_sessions()
    assert len(sessions) == 2
    # b (last_active 11:00) should come first
    assert sessions[0].session_id == "b"
    assert sessions[1].session_id == "a"


@pytest.mark.asyncio
async def test_list_sessions_empty(registry, mock_redis):
    """list_sessions() should return [] when no keys found."""
    async def fake_scan_iter(**kwargs):
        return
        yield  # make it an async generator
    mock_redis.scan_iter = fake_scan_iter
    sessions = await registry.list_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_delete_existing_session(registry, mock_redis):
    """delete() should call Redis DELETE and return True."""
    mock_redis.delete = AsyncMock(return_value=1)
    result = await registry.delete("sess-x")
    assert result is True
    mock_redis.delete.assert_called_once_with("goose:session:sess-x")


@pytest.mark.asyncio
async def test_delete_nonexistent_session(registry, mock_redis):
    """delete() should return False when key did not exist."""
    mock_redis.delete = AsyncMock(return_value=0)
    result = await registry.delete("no-such")
    assert result is False
