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
    mock_registry = AsyncMock()
    mock_registry.register = AsyncMock()
    with patch("life_core.goose_api._get_client") as mock_get, \
         patch("life_core.goose_api._get_registry", return_value=mock_registry):
        mock_client = AsyncMock()
        mock_session = MagicMock(session_id="new-session", working_dir="workspace")
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_get.return_value = mock_client
        resp = client.post("/goose/sessions", json={"working_dir": "workspace"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "new-session"
    mock_registry.register.assert_called_once_with("new-session", "workspace")


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


def test_goose_sessions_list(client):
    """GET /goose/sessions returns list from registry."""
    from life_core.goose_sessions import SessionInfo
    fake_sessions = [
        SessionInfo("s1", "proj", "2026-04-07T10:00:00+00:00", "2026-04-07T11:00:00+00:00", 3),
        SessionInfo("s2", ".", "2026-04-07T09:00:00+00:00", "2026-04-07T09:30:00+00:00", 1),
    ]
    mock_registry = AsyncMock()
    mock_registry.list_sessions = AsyncMock(return_value=fake_sessions)
    with patch("life_core.goose_api._get_registry", return_value=mock_registry):
        resp = client.get("/goose/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sessions"]) == 2
    assert data["sessions"][0]["session_id"] == "s1"


def test_goose_session_delete(client):
    """DELETE /goose/sessions/{id} returns 204 when session exists."""
    mock_registry = AsyncMock()
    mock_registry.delete = AsyncMock(return_value=True)
    with patch("life_core.goose_api._get_registry", return_value=mock_registry):
        resp = client.delete("/goose/sessions/s1")
    assert resp.status_code == 204
    mock_registry.delete.assert_called_once_with("s1")


def test_goose_session_delete_not_found(client):
    """DELETE /goose/sessions/{id} returns 404 when session does not exist."""
    mock_registry = AsyncMock()
    mock_registry.delete = AsyncMock(return_value=False)
    with patch("life_core.goose_api._get_registry", return_value=mock_registry):
        resp = client.delete("/goose/sessions/no-such")
    assert resp.status_code == 404


def test_goose_session_resume(client):
    with patch("life_core.goose_api._get_client") as mock_gc, \
         patch("life_core.goose_api._get_registry") as mock_gr:
        mock_client = AsyncMock()
        mock_session = MagicMock(session_id="resumed-s1")
        mock_client.load_session = AsyncMock(return_value=mock_session)
        mock_gc.return_value = mock_client
        mock_reg = AsyncMock()
        mock_reg.touch = AsyncMock()
        mock_gr.return_value = mock_reg
        resp = client.post("/goose/sessions/resumed-s1/resume")
    assert resp.status_code == 200
    assert resp.json()["resumed"] is True


def test_goose_prompt_streams(client):
    async def fake_prompt(session_id, text):
        yield {"jsonrpc": "2.0", "method": "AgentMessageChunk", "params": {"content": "hi"}}

    mock_registry = AsyncMock()
    mock_registry.touch = AsyncMock()
    with patch("life_core.goose_api._get_client") as mock_get, \
         patch("life_core.goose_api._get_registry", return_value=mock_registry):
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
