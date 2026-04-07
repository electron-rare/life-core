"""Redis-backed session registry for Goose agent sessions."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import redis.asyncio as aioredis

_TTL_SECONDS = 24 * 3600  # 24 hours


@dataclass
class SessionInfo:
    session_id: str
    working_dir: str
    created_at: str
    last_active: str
    message_count: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(session_id: str) -> str:
    return f"goose:session:{session_id}"


def _decode(raw: dict) -> SessionInfo:
    """Decode a Redis hash (bytes keys/values) into a SessionInfo."""
    def _v(k: str) -> str:
        val = raw.get(k.encode()) or raw.get(k, b"")
        return val.decode() if isinstance(val, bytes) else str(val)

    return SessionInfo(
        session_id=_v("session_id"),
        working_dir=_v("working_dir"),
        created_at=_v("created_at"),
        last_active=_v("last_active"),
        message_count=int(_v("message_count") or 0),
    )


class SessionRegistry:
    """Manages Goose session metadata in Redis."""

    def __init__(self) -> None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis = aioredis.from_url(redis_url, decode_responses=False)

    async def register(self, session_id: str, working_dir: str) -> SessionInfo:
        """Store a new session and return its SessionInfo. TTL: 24 h."""
        now = _now_iso()
        info = SessionInfo(
            session_id=session_id,
            working_dir=working_dir,
            created_at=now,
            last_active=now,
            message_count=0,
        )
        k = _key(session_id)
        mapping = {field: str(val) for field, val in asdict(info).items()}
        await self._redis.hset(k, mapping=mapping)
        await self._redis.expire(k, _TTL_SECONDS)
        return info

    async def touch(self, session_id: str) -> None:
        """Update last_active, increment message_count, reset TTL."""
        k = _key(session_id)
        raw = await self._redis.hgetall(k)
        if not raw:
            return
        await self._redis.hset(k, "last_active", _now_iso())
        await self._redis.hincrby(k, "message_count", 1)
        await self._redis.expire(k, _TTL_SECONDS)

    async def list_sessions(self) -> list[SessionInfo]:
        """Return all sessions sorted by last_active descending."""
        keys: list[bytes] = []
        async for k in self._redis.scan_iter(match="goose:session:*", count=100):
            keys.append(k)
        sessions: list[SessionInfo] = []
        for k in keys:
            raw = await self._redis.hgetall(k)
            if raw:
                sessions.append(_decode(raw))
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        return sessions

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed, False otherwise."""
        deleted = await self._redis.delete(_key(session_id))
        return deleted > 0
