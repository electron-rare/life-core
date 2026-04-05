"""Tests for conversations CRUD API."""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from life_core.conversations_api import conversations_router, reset_conversation_store, set_redis


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
    reset_conversation_store()
    set_redis(fake_redis)
    yield TestClient(app)
    set_redis(None)
    reset_conversation_store()


def test_list_empty(client):
    response = client.get("/conversations")
    assert response.status_code == 200
    assert response.json()["conversations"] == []


def test_create_conversation(client):
    response = client.post("/conversations", json={"title": "Test conv", "provider": "ollama"})
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test conv"
    assert data["provider"] == "ollama"
    assert "id" in data


def test_create_conversation_defaults_to_auto_provider(client):
    response = client.post("/conversations", json={"title": "Auto provider"})
    assert response.status_code == 200
    assert response.json()["provider"] == "auto"


def test_get_conversation(client):
    create_resp = client.post("/conversations", json={"title": "Get test"})
    conv_id = create_resp.json()["id"]

    response = client.get(f"/conversations/{conv_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Get test"


def test_add_message(client):
    create_resp = client.post("/conversations", json={"title": "Msg test"})
    conv_id = create_resp.json()["id"]

    response = client.post(f"/conversations/{conv_id}/messages", json={"role": "user", "content": "Hello"})
    assert response.status_code == 200
    assert response.json()["message_count"] == 1

    response = client.post(f"/conversations/{conv_id}/messages", json={"role": "assistant", "content": "Hi!"})
    assert response.json()["message_count"] == 2


def test_delete_conversation(client):
    create_resp = client.post("/conversations", json={"title": "Delete test"})
    conv_id = create_resp.json()["id"]

    response = client.delete(f"/conversations/{conv_id}")
    assert response.status_code == 200

    response = client.get(f"/conversations/{conv_id}")
    assert response.status_code == 404


def test_get_not_found(client):
    response = client.get("/conversations/nonexistent")
    assert response.status_code == 404


def test_falls_back_to_in_memory_store_without_redis(client):
    set_redis(None)
    reset_conversation_store()

    create_resp = client.post("/conversations", json={"title": "Fallback test"})
    assert create_resp.status_code == 200
    conv_id = create_resp.json()["id"]

    message_resp = client.post(
        f"/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Hello fallback"},
    )
    assert message_resp.status_code == 200
    assert message_resp.json()["message_count"] == 1

    response = client.get("/conversations")
    assert response.status_code == 200
    assert response.json()["conversations"][0]["id"] == conv_id
