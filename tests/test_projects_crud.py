"""Tests for project CRUD, task endpoints, team, and sync."""
from __future__ import annotations

import json as json_module
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import life_core.projects.router as projects_router_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis():
    r = MagicMock()
    r.get = MagicMock(return_value=None)
    r.set = MagicMock(return_value=True)
    r.delete = MagicMock(return_value=1)
    r.hset = MagicMock(return_value=1)
    r.hgetall = MagicMock(return_value={})

    async def _scan_iter(pattern):
        return
        yield  # make it an async generator

    r.scan_iter = _scan_iter
    return r


def _app_with_redis(redis_mock):
    """Build a minimal FastAPI test app with just projects + team routers."""
    from fastapi import FastAPI
    from life_core.projects.router import router as proj_router, team_router

    projects_router_module._redis = redis_mock

    app = FastAPI()
    app.include_router(proj_router)
    app.include_router(team_router)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_project():
    r = _make_redis()
    app = _app_with_redis(r)
    client = TestClient(app)

    resp = client.post("/projects", json={"name": "alpha", "client": "ACME"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "alpha"
    assert data["client"] == "ACME"
    assert r.set.called


def test_update_project():
    r = _make_redis()
    existing = {"name": "alpha", "client": "ACME", "repo": "", "hardware": {}, "firmware": {}, "agents": []}
    r.get = MagicMock(return_value=json_module.dumps(existing).encode())
    app = _app_with_redis(r)
    client = TestClient(app)

    resp = client.put("/projects/alpha", json={"client": "NewClient"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["client"] == "NewClient"


def test_create_task():
    r = _make_redis()
    app = _app_with_redis(r)
    client = TestClient(app)

    resp = client.post("/projects/alpha/tasks", json={"name": "Design PCB", "gate": "s1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Design PCB"
    assert data["gate"] == "s1"
    assert "id" in data


def test_list_tasks_empty():
    r = _make_redis()
    app = _app_with_redis(r)
    client = TestClient(app)

    resp = client.get("/projects/alpha/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"] == []
    assert data["count"] == 0


def test_team_members():
    r = _make_redis()
    app = _app_with_redis(r)
    client = TestClient(app)

    with patch("life_core.projects.team.get_team_members", new=AsyncMock(return_value=[])):
        resp = client.get("/team/members")
    assert resp.status_code == 200
    data = resp.json()
    assert "members" in data


def test_sync_project_no_github_token():
    """Sync with no GitHub token — should attempt fetch and fail with 502."""
    r = _make_redis()
    existing = {"name": "alpha", "client": "ACME"}
    r.get = MagicMock(return_value=json_module.dumps(existing).encode())
    app = _app_with_redis(r)
    client = TestClient(app)

    import httpx

    async def _fake_fetch(name):
        raise httpx.HTTPStatusError(
            "not found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

    with patch("life_core.projects.router.fetch_remote_yaml", new=_fake_fetch):
        async def _fake_push(name, content, sha, message):
            raise httpx.HTTPStatusError(
                "unauthorized",
                request=MagicMock(),
                response=MagicMock(status_code=401),
            )

        with patch("life_core.projects.router.push_yaml", new=_fake_push):
            resp = client.post("/projects/alpha/sync")

    assert resp.status_code == 502
