"""Extended tests for projects/router.py — cache, YAML parsing, GitHub fetch, CRUD, tasks, timeline."""
from __future__ import annotations

import json as json_module
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import life_core.projects.router as router_mod
from life_core.projects.router import (
    router as proj_router,
    team_router,
    _parse_gate_status,
    _parse_kill_life_yaml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redis():
    """Create a mock Redis client with common methods."""
    r = MagicMock()
    r.get = MagicMock(return_value=None)
    r.set = MagicMock(return_value=True)
    r.delete = MagicMock(return_value=1)
    r.hset = MagicMock(return_value=1)
    r.hgetall = MagicMock(return_value={})

    async def _scan_iter(pattern, **kwargs):
        return
        yield  # async generator

    r.scan_iter = _scan_iter
    return r


def _app(redis_mock):
    router_mod._redis = redis_mock
    app = FastAPI()
    app.include_router(proj_router)
    app.include_router(team_router)
    return app


# ---------------------------------------------------------------------------
# Pure functions: _parse_gate_status
# ---------------------------------------------------------------------------


class TestParseGateStatus:
    def test_all_gates_present(self):
        gates = {
            "s0": {"status": "passed", "date": "2026-01-01"},
            "s1": {"status": "in_progress"},
            "s2": {},
            "s3": {"status": "blocked", "date": "2026-03-15"},
        }
        result = _parse_gate_status(gates)
        assert result["s0"]["status"] == "passed"
        assert result["s0"]["date"] == "2026-01-01"
        assert result["s1"]["status"] == "in_progress"
        assert result["s1"]["date"] is None
        assert result["s2"]["status"] == "pending"
        assert result["s3"]["status"] == "blocked"

    def test_empty_gates(self):
        result = _parse_gate_status({})
        for gate in ("s0", "s1", "s2", "s3"):
            assert result[gate]["status"] == "pending"
            assert result[gate]["date"] is None

    def test_non_dict_gate_value(self):
        gates = {"s0": "some_string", "s1": 42, "s2": None, "s3": True}
        result = _parse_gate_status(gates)
        for gate in ("s0", "s1", "s2", "s3"):
            assert result[gate]["status"] == "pending"

    def test_extra_gates_ignored(self):
        gates = {"s0": {"status": "done"}, "s4": {"status": "extra"}}
        result = _parse_gate_status(gates)
        assert "s4" not in result
        assert result["s0"]["status"] == "done"


# ---------------------------------------------------------------------------
# Pure functions: _parse_kill_life_yaml
# ---------------------------------------------------------------------------


class TestParseKillLifeYaml:
    def test_valid_yaml(self):
        content = """
kill_life:
  project: my-pcb
  repo: L-electron-Rare/my-pcb
  client: ACME
  gates:
    s0:
      status: passed
      date: "2026-01-15"
    s1:
      status: in_progress
  hardware:
    pcb_dir: hardware/pcb
  firmware:
    framework: platformio
"""
        result = _parse_kill_life_yaml(content, "my-pcb.yaml")
        assert result is not None
        assert result["name"] == "my-pcb"
        assert result["repo"] == "L-electron-Rare/my-pcb"
        assert result["client"] == "ACME"
        assert result["gates"]["s0"]["status"] == "passed"
        assert result["hardware"]["pcb_dir"] == "hardware/pcb"

    def test_minimal_yaml(self):
        content = "kill_life:\n  project: minimal\n"
        result = _parse_kill_life_yaml(content, "minimal.yaml")
        assert result is not None
        assert result["name"] == "minimal"
        assert result["gates"]["s0"]["status"] == "pending"

    def test_no_kill_life_key_uses_root(self):
        content = "project: direct\nrepo: some/repo\n"
        result = _parse_kill_life_yaml(content, "direct.yaml")
        assert result is not None
        assert result["name"] == "direct"

    def test_project_name_from_filename(self):
        content = "kill_life:\n  repo: some/repo\n"
        result = _parse_kill_life_yaml(content, "fallback-name.yaml")
        assert result is not None
        assert result["name"] == "fallback-name"

    def test_invalid_yaml_returns_none(self):
        content = "{{invalid: yaml: :::"
        result = _parse_kill_life_yaml(content, "bad.yaml")
        assert result is None

    def test_non_dict_returns_none(self):
        content = "- item1\n- item2\n"
        result = _parse_kill_life_yaml(content, "list.yaml")
        assert result is None

    def test_non_dict_kill_life_returns_none(self):
        content = "kill_life: just_a_string\n"
        result = _parse_kill_life_yaml(content, "string.yaml")
        assert result is None

    def test_gates_as_non_dict_treated_as_empty(self):
        content = "kill_life:\n  project: test\n  gates: not_a_dict\n"
        result = _parse_kill_life_yaml(content, "test.yaml")
        assert result is not None
        assert result["gates"]["s0"]["status"] == "pending"


# ---------------------------------------------------------------------------
# Cache functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_get_returns_none_when_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        from life_core.projects.router import _cache_get
        result = await _cache_get("any-key")
        assert result is None
    finally:
        router_mod._redis = old_redis


@pytest.mark.asyncio
async def test_cache_get_returns_data():
    r = _make_redis()
    r.get = MagicMock(return_value=json_module.dumps([{"name": "test"}]).encode())
    old_redis = router_mod._redis
    try:
        router_mod._redis = r
        from life_core.projects.router import _cache_get
        result = await _cache_get("projects:all")
        assert result == [{"name": "test"}]
    finally:
        router_mod._redis = old_redis


@pytest.mark.asyncio
async def test_cache_get_handles_exception():
    r = MagicMock()
    r.get = MagicMock(side_effect=Exception("redis error"))
    old_redis = router_mod._redis
    try:
        router_mod._redis = r
        from life_core.projects.router import _cache_get
        result = await _cache_get("any-key")
        assert result is None
    finally:
        router_mod._redis = old_redis


@pytest.mark.asyncio
async def test_cache_set_stores_data():
    r = _make_redis()
    old_redis = router_mod._redis
    try:
        router_mod._redis = r
        from life_core.projects.router import _cache_set
        await _cache_set("projects:all", [{"name": "test"}])
        r.set.assert_called_once()
        call_args = r.set.call_args
        assert call_args[0][0] == "projects:all"
        assert json_module.loads(call_args[0][1]) == [{"name": "test"}]
    finally:
        router_mod._redis = old_redis


@pytest.mark.asyncio
async def test_cache_set_noop_when_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        from life_core.projects.router import _cache_set
        await _cache_set("key", "value")  # should not raise
    finally:
        router_mod._redis = old_redis


@pytest.mark.asyncio
async def test_cache_set_handles_exception():
    r = MagicMock()
    r.set = MagicMock(side_effect=Exception("redis error"))
    old_redis = router_mod._redis
    try:
        router_mod._redis = r
        from life_core.projects.router import _cache_set
        await _cache_set("key", "value")  # should not raise
    finally:
        router_mod._redis = old_redis


# ---------------------------------------------------------------------------
# GitHub headers
# ---------------------------------------------------------------------------


def test_github_headers_with_token():
    from life_core.projects.router import _github_headers
    with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
        headers = _github_headers()
    assert headers["Authorization"] == "Bearer ghp_test123"
    assert "application/vnd.github" in headers["Accept"]


def test_github_headers_without_token():
    from life_core.projects.router import _github_headers
    with patch.dict("os.environ", {}, clear=True):
        headers = _github_headers()
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# _fetch_projects_from_github
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_projects_404():
    from life_core.projects.router import _fetch_projects_from_github

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.router.httpx.AsyncClient", return_value=mock_client):
        result = await _fetch_projects_from_github()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_projects_403():
    from life_core.projects.router import _fetch_projects_from_github

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.router.httpx.AsyncClient", return_value=mock_client):
        result = await _fetch_projects_from_github()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_projects_success():
    from life_core.projects.router import _fetch_projects_from_github
    import base64

    yaml_content = "kill_life:\n  project: test-proj\n  repo: org/test\n"
    dir_response = MagicMock()
    dir_response.status_code = 200
    dir_response.raise_for_status = MagicMock()
    dir_response.json.return_value = [
        {"name": "test-proj.yaml", "download_url": "https://raw.github.com/test.yaml"},
        {"name": "README.md", "download_url": "https://raw.github.com/readme"},
    ]

    file_response = MagicMock()
    file_response.raise_for_status = MagicMock()
    file_response.text = yaml_content

    call_count = 0

    async def fake_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return dir_response
        return file_response

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.router.httpx.AsyncClient", return_value=mock_client):
        result = await _fetch_projects_from_github()
    assert len(result) == 1
    assert result[0]["name"] == "test-proj"


@pytest.mark.asyncio
async def test_fetch_projects_non_list_response():
    from life_core.projects.router import _fetch_projects_from_github

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": "not a list"}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.router.httpx.AsyncClient", return_value=mock_client):
        result = await _fetch_projects_from_github()
    assert result == []


# ---------------------------------------------------------------------------
# list_projects / get_project endpoints
# ---------------------------------------------------------------------------


def test_list_projects_from_cache():
    r = _make_redis()
    cached_projects = [{"name": "proj1", "gates": {}}]
    r.get = MagicMock(return_value=json_module.dumps(cached_projects).encode())
    app = _app(r)
    client = TestClient(app)

    resp = client.get("/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["projects"][0]["name"] == "proj1"


def test_get_project_found():
    r = _make_redis()
    cached_projects = [
        {"name": "alpha", "repo": "org/alpha"},
        {"name": "beta", "repo": "org/beta"},
    ]
    r.get = MagicMock(return_value=json_module.dumps(cached_projects).encode())
    app = _app(r)
    client = TestClient(app)

    resp = client.get("/projects/alpha")
    assert resp.status_code == 200
    assert resp.json()["name"] == "alpha"


def test_get_project_not_found():
    r = _make_redis()
    cached_projects = [{"name": "alpha"}]
    r.get = MagicMock(return_value=json_module.dumps(cached_projects).encode())
    app = _app(r)
    client = TestClient(app)

    resp = client.get("/projects/nonexistent")
    assert resp.status_code == 404


def test_list_projects_github_error():
    r = _make_redis()
    r.get = MagicMock(return_value=None)
    app = _app(r)
    client = TestClient(app)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("network error"))

    with patch("life_core.projects.router.httpx.AsyncClient", return_value=mock_client):
        resp = client.get("/projects")

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


def test_create_project_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.post("/projects", json={"name": "test"})
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


def test_create_project_with_all_fields():
    r = _make_redis()
    app = _app(r)
    client = TestClient(app)

    resp = client.post("/projects", json={
        "name": "full-project",
        "client": "BigCorp",
        "repo": "org/full-project",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "full-project"
    assert data["client"] == "BigCorp"
    assert "hardware" in data
    assert "firmware" in data
    assert "agents" in data


def test_update_project_existing():
    r = _make_redis()
    existing = {"name": "alpha", "client": "Old", "repo": ""}
    r.get = MagicMock(return_value=json_module.dumps(existing).encode())
    app = _app(r)
    client = TestClient(app)

    resp = client.put("/projects/alpha", json={"client": "NewClient", "repo": "org/alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["client"] == "NewClient"
    assert data["repo"] == "org/alpha"


def test_update_project_with_gates():
    r = _make_redis()
    existing = {"name": "alpha", "client": "ACME", "gates": {}}
    r.get = MagicMock(return_value=json_module.dumps(existing).encode())
    app = _app(r)
    client = TestClient(app)

    resp = client.put("/projects/alpha", json={
        "gates": {"s0": {"status": "passed", "date": "2026-01-01"}}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["gates"]["s0"]["status"] == "passed"


def test_update_project_not_in_redis_fallback():
    """When project not in Redis, fallback creates minimal entry."""
    r = _make_redis()
    r.get = MagicMock(return_value=None)
    app = _app(r)
    client = TestClient(app)

    with patch("life_core.projects.router.fetch_remote_yaml", new=AsyncMock(side_effect=Exception("not found"))):
        resp = client.put("/projects/new-proj", json={"client": "New"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "new-proj"
    assert data["client"] == "New"


def test_delete_project_success():
    r = _make_redis()
    r.delete = MagicMock(return_value=1)
    app = _app(r)
    client = TestClient(app)

    resp = client.delete("/projects/alpha")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "alpha"


def test_delete_project_not_found():
    r = _make_redis()
    r.delete = MagicMock(return_value=0)
    app = _app(r)
    client = TestClient(app)

    resp = client.delete("/projects/nonexistent")
    assert resp.status_code == 404


def test_delete_project_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.delete("/projects/alpha")
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


# ---------------------------------------------------------------------------
# Task endpoints via API
# ---------------------------------------------------------------------------


def test_list_tasks_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.get("/projects/alpha/tasks")
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


def test_create_task_via_api():
    r = _make_redis()
    app = _app(r)
    client = TestClient(app)

    resp = client.post("/projects/alpha/tasks", json={"name": "Wire ERC", "gate": "s1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Wire ERC"
    assert data["gate"] == "s1"
    assert "id" in data


def test_create_task_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.post("/projects/alpha/tasks", json={"name": "test", "gate": "s0"})
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


def test_update_task_via_api():
    r = _make_redis()
    import json as json_mod
    task_hash = {
        b"id": b"abcd1234",
        b"name": b"Wire ERC",
        b"gate": b"s1",
        b"assignees": json_mod.dumps([]).encode(),
        b"start_date": b"",
        b"end_date": b"",
        b"depends_on": json_mod.dumps([]).encode(),
        b"status": b"todo",
        b"progress": b"0",
    }
    r.hgetall = MagicMock(return_value=task_hash)
    app = _app(r)
    client = TestClient(app)

    resp = client.put("/projects/alpha/tasks/abcd1234", json={"status": "done", "progress": 100})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["progress"] == 100


def test_update_task_not_found():
    r = _make_redis()
    r.hgetall = MagicMock(return_value={})
    app = _app(r)
    client = TestClient(app)

    resp = client.put("/projects/alpha/tasks/missing", json={"status": "done"})
    assert resp.status_code == 404


def test_update_task_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.put("/projects/alpha/tasks/abc", json={"status": "done"})
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


def test_delete_task_via_api():
    r = _make_redis()
    r.delete = MagicMock(return_value=1)
    app = _app(r)
    client = TestClient(app)

    resp = client.delete("/projects/alpha/tasks/abcd1234")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "abcd1234"


def test_delete_task_not_found():
    r = _make_redis()
    r.delete = MagicMock(return_value=0)
    app = _app(r)
    client = TestClient(app)

    resp = client.delete("/projects/alpha/tasks/missing")
    assert resp.status_code == 404


def test_delete_task_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.delete("/projects/alpha/tasks/abc")
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


# ---------------------------------------------------------------------------
# Timeline endpoint
# ---------------------------------------------------------------------------


def test_timeline_empty():
    r = _make_redis()
    app = _app(r)
    client = TestClient(app)

    resp = client.get("/projects/alpha/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["timeline"] == []


def test_timeline_no_redis():
    old_redis = router_mod._redis
    try:
        router_mod._redis = None
        app = FastAPI()
        app.include_router(proj_router)
        client = TestClient(app)
        resp = client.get("/projects/alpha/timeline")
        assert resp.status_code == 503
    finally:
        router_mod._redis = old_redis


# ---------------------------------------------------------------------------
# Sync endpoint
# ---------------------------------------------------------------------------


def test_sync_project_not_in_redis():
    r = _make_redis()
    r.get = MagicMock(return_value=None)
    app = _app(r)
    client = TestClient(app)

    resp = client.post("/projects/missing/sync")
    assert resp.status_code == 404


def test_sync_project_success():
    r = _make_redis()
    existing = {"name": "alpha", "client": "ACME"}
    r.get = MagicMock(return_value=json_module.dumps(existing).encode())
    app = _app(r)
    client = TestClient(app)

    with patch("life_core.projects.router.fetch_remote_yaml", new=AsyncMock(return_value=("content", "sha123"))):
        with patch("life_core.projects.router.push_yaml", new=AsyncMock(return_value="commit_abc")):
            resp = client.post("/projects/alpha/sync")

    assert resp.status_code == 200
    data = resp.json()
    assert data["synced"] == "alpha"
    assert data["commit"] == "commit_abc"


# ---------------------------------------------------------------------------
# Team endpoint
# ---------------------------------------------------------------------------


def test_team_members_returns_agents():
    r = _make_redis()
    app = _app(r)
    client = TestClient(app)

    from life_core.projects.models import TeamMember
    agents = [
        TeamMember(id="forge", name="Forge", type="agent"),
        TeamMember(id="goose", name="Goose", type="agent"),
    ]

    with patch("life_core.projects.router.get_team_members", new=AsyncMock(return_value=agents)):
        resp = client.get("/team/members")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["members"][0]["id"] == "forge"
