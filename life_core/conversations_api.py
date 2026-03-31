"""Conversations API — CRUD for chat conversations in Redis."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("life_core.conversations")

conversations_router = APIRouter(prefix="/conversations", tags=["Conversations"])

_redis = None


def set_redis(redis_client) -> None:
    global _redis
    _redis = redis_client


def _get_redis():
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return _redis


CONV_PREFIX = "conv:"


class ConversationCreate(BaseModel):
    title: str = "Nouvelle conversation"
    provider: str = "ollama"


class MessageAdd(BaseModel):
    role: str
    content: str


@conversations_router.get("")
async def list_conversations():
    """List all conversations."""
    r = _get_redis()
    keys = r.keys(f"{CONV_PREFIX}*")
    conversations = []
    for key in sorted(keys, reverse=True)[:50]:
        data = r.get(key)
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
    r = _get_redis()
    conv_id = str(uuid.uuid4())[:8]
    conv = {
        "id": conv_id,
        "title": body.title,
        "provider": body.provider,
        "messages": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    r.set(f"{CONV_PREFIX}{conv_id}", json.dumps(conv), ex=86400 * 30)  # 30 days TTL
    return conv


@conversations_router.get("/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a conversation by ID."""
    r = _get_redis()
    data = r.get(f"{CONV_PREFIX}{conv_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return json.loads(data)


@conversations_router.post("/{conv_id}/messages")
async def add_message(conv_id: str, msg: MessageAdd):
    """Add a message to a conversation."""
    r = _get_redis()
    data = r.get(f"{CONV_PREFIX}{conv_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = json.loads(data)
    conv["messages"].append({"role": msg.role, "content": msg.content})
    r.set(f"{CONV_PREFIX}{conv_id}", json.dumps(conv), ex=86400 * 30)
    return {"status": "ok", "message_count": len(conv["messages"])}


@conversations_router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation."""
    r = _get_redis()
    deleted = r.delete(f"{CONV_PREFIX}{conv_id}")
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}
