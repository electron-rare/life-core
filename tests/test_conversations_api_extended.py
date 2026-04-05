"""Extended tests for conversations CRUD API — edge cases and additional paths."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.conversations_api import conversations_router, set_redis


class FakeRedis:
    """In-memory Redis mock."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        if key in self.store:
            del self.store[key]
            return 1
        return 0

    def keys(self, pattern="*"):
        import fnmatch
        return [k for k in self.store if fnmatch.fnmatch(k, pattern)]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(conversations_router)
    fake_redis = FakeRedis()
    set_redis(fake_redis)
    yield TestClient(app)
    set_redis(None)


# ---------------------------------------------------------------------------
# list conversations
# ---------------------------------------------------------------------------


def test_list_multiple_conversations(client):
    client.post("/conversations", json={"title": "Conv A"})
    client.post("/conversations", json={"title": "Conv B"})
    response = client.get("/conversations")
    data = response.json()
    assert len(data["conversations"]) == 2


def test_list_conversation_has_required_fields(client):
    client.post("/conversations", json={"title": "Test"})
    data = client.get("/conversations").json()
    conv = data["conversations"][0]
    assert "id" in conv
    assert "title" in conv
    assert "created_at" in conv
    assert "provider" in conv
    assert "message_count" in conv


def test_list_shows_message_count(client):
    resp = client.post("/conversations", json={"title": "With msgs"})
    conv_id = resp.json()["id"]
    client.post(f"/conversations/{conv_id}/messages", json={"role": "user", "content": "hi"})
    client.post(f"/conversations/{conv_id}/messages", json={"role": "assistant", "content": "hello"})

    data = client.get("/conversations").json()
    conv = next(c for c in data["conversations"] if c["id"] == conv_id)
    assert conv["message_count"] == 2


# ---------------------------------------------------------------------------
# create conversation
# ---------------------------------------------------------------------------


def test_create_returns_id(client):
    resp = client.post("/conversations", json={"title": "New"})
    assert "id" in resp.json()
    assert len(resp.json()["id"]) > 0


def test_create_default_provider_is_auto(client):
    resp = client.post("/conversations", json={"title": "Default provider"})
    assert resp.json()["provider"] == "auto"


def test_create_sets_empty_messages(client):
    resp = client.post("/conversations", json={"title": "Empty"})
    assert resp.json()["messages"] == []


def test_create_stores_created_at(client):
    resp = client.post("/conversations", json={"title": "Timed"})
    assert "created_at" in resp.json()
    assert "T" in resp.json()["created_at"]


# ---------------------------------------------------------------------------
# add message
# ---------------------------------------------------------------------------


def test_add_message_to_nonexistent_conversation_returns_404(client):
    response = client.post(
        "/conversations/nonexistent-id/messages",
        json={"role": "user", "content": "hello"},
    )
    assert response.status_code == 404


def test_add_message_increments_count(client):
    resp = client.post("/conversations", json={"title": "Messages"})
    conv_id = resp.json()["id"]

    for i in range(3):
        r = client.post(f"/conversations/{conv_id}/messages",
                        json={"role": "user", "content": f"msg {i}"})
        assert r.json()["message_count"] == i + 1


def test_add_message_persists_content(client):
    resp = client.post("/conversations", json={"title": "Persist"})
    conv_id = resp.json()["id"]
    client.post(f"/conversations/{conv_id}/messages",
                json={"role": "user", "content": "persistent message"})

    conv = client.get(f"/conversations/{conv_id}").json()
    assert any(m["content"] == "persistent message" for m in conv["messages"])


# ---------------------------------------------------------------------------
# delete conversation
# ---------------------------------------------------------------------------


def test_delete_nonexistent_returns_404(client):
    response = client.delete("/conversations/ghost-id")
    assert response.status_code == 404


def test_delete_returns_deleted_status(client):
    resp = client.post("/conversations", json={"title": "Delete me"})
    conv_id = resp.json()["id"]
    delete_resp = client.delete(f"/conversations/{conv_id}")
    assert delete_resp.json()["status"] == "deleted"
