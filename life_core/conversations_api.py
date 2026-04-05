"""Conversations API — CRUD for chat conversations in Redis."""

from __future__ import annotations

import fnmatch
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("life_core.conversations")

conversations_router = APIRouter(prefix="/conversations", tags=["Conversations"])

_redis = None
_fallback_warning_emitted = False


class _InMemoryConversationStore:
    """Ephemeral fallback store used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def clear(self) -> None:
        self._store.clear()

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    def keys(self, pattern: str = "*") -> list[str]:
        return [key for key in self._store if fnmatch.fnmatch(key, pattern)]


_fallback_store = _InMemoryConversationStore()


def set_redis(redis_client) -> None:
    global _redis, _fallback_warning_emitted
    _redis = redis_client
    if redis_client is not None:
        _fallback_warning_emitted = False


def reset_conversation_store() -> None:
    _fallback_store.clear()


def _get_store():
    global _fallback_warning_emitted
    if _redis is not None:
        return _redis
    if not _fallback_warning_emitted:
        logger.warning("Redis not available, using in-memory conversation store")
        _fallback_warning_emitted = True
    return _fallback_store


CONV_PREFIX = "conv:"


class ConversationCreate(BaseModel):
    title: str = "Nouvelle conversation"
    provider: str = "auto"


class MessageAdd(BaseModel):
    role: str
    content: str


@conversations_router.get("")
async def list_conversations():
    """List all conversations."""
    store = _get_store()
    keys = store.keys(f"{CONV_PREFIX}*")
    conversations = []
    for key in sorted(keys, reverse=True)[:50]:
        data = store.get(key)
        if data:
            conv = json.loads(data)
            conversations.append({
                "id": conv["id"],
                "title": conv.get("title", "Sans titre"),
                "created_at": conv.get("created_at", ""),
                "provider": conv.get("provider", ""),
                "message_count": len(conv.get("messages", [])),
            })
    return {"conversations": conversations}


@conversations_router.post("")
async def create_conversation(body: ConversationCreate):
    """Create a new conversation."""
    store = _get_store()
    conv_id = str(uuid.uuid4())[:8]
    conv = {
        "id": conv_id,
        "title": body.title,
        "provider": body.provider,
        "messages": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    store.set(f"{CONV_PREFIX}{conv_id}", json.dumps(conv), ex=86400 * 30)  # 30 days TTL
    return conv


@conversations_router.get("/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a conversation by ID."""
    store = _get_store()
    data = store.get(f"{CONV_PREFIX}{conv_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return json.loads(data)


@conversations_router.post("/{conv_id}/messages")
async def add_message(conv_id: str, msg: MessageAdd):
    """Add a message to a conversation."""
    store = _get_store()
    data = store.get(f"{CONV_PREFIX}{conv_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = json.loads(data)
    conv["messages"].append({"role": msg.role, "content": msg.content})
    store.set(f"{CONV_PREFIX}{conv_id}", json.dumps(conv), ex=86400 * 30)
    return {"status": "ok", "message_count": len(conv["messages"])}


@conversations_router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    store = _get_store()
    deleted = store.delete(f"{CONV_PREFIX}{conv_id}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}
