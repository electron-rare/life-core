"""Tests for Goose API router."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.goose_api import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_goose_health_ok(client):
    with patch("life_core.goose_api._get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.health = AsyncMock(return_value={"status": "ok"})
        mock_get.return_value = mock_client
        resp = client.get("/goose/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_goose_health_down(client):
    with patch("life_core.goose_api._get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.health = AsyncMock(side_effect=Exception("Connection refused"))
        mock_get.return_value = mock_client
        resp = client.get("/goose/health")
    assert resp.status_code == 502


def test_goose_recipes_list(client):
    resp = client.get("/goose/recipes")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["recipes"], list)
    names = [r["name"] for r in data["recipes"]]
    assert "review-kicad" in names


def test_goose_session_create(client):
    with patch("life_core.goose_api._get_client") as mock_get:
        mock_client = AsyncMock()
        mock_session = MagicMock(session_id="new-session", working_dir="workspace")
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_get.return_value = mock_client
        resp = client.post("/goose/sessions", json={"working_dir": "workspace"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "new-session"


def test_goose_session_rejects_absolute_path(client):
    resp = client.post("/goose/sessions", json={"working_dir": "/etc/passwd"})
    assert resp.status_code == 400


def test_goose_session_rejects_traversal(client):
    resp = client.post("/goose/sessions", json={"working_dir": "../../etc"})
    assert resp.status_code == 400


def test_goose_recipe_run(client):
    with patch("life_core.goose_api._get_client") as mock_get:
        mock_client = AsyncMock()
        mock_session = MagicMock(session_id="recipe-session")
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.prompt_sync = AsyncMock(return_value="done")
        mock_get.return_value = mock_client
        resp = client.post("/goose/recipes/debug-infra/run", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) >= 2
    assert all(r["status"] == "ok" for r in data["results"])


def test_goose_prompt_streams(client):
    async def fake_prompt(session_id, text):
        yield {"jsonrpc": "2.0", "method": "AgentMessageChunk", "params": {"content": "hi"}}

    with patch("life_core.goose_api._get_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.prompt = fake_prompt
        mock_get.return_value = mock_client
        resp = client.post(
            "/goose/prompt",
            json={"session_id": "s1", "prompt": "hello"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "AgentMessageChunk" in resp.text
