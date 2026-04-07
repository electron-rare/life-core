"""Tests for TaskStore Redis CRUD."""
from __future__ import annotations

import json as json_module
from unittest.mock import AsyncMock, MagicMock

import pytest

from life_core.projects.models import TaskCreate
from life_core.projects.task_store import TaskStore


def _make_redis_mock():
    r = MagicMock()
    r.hset = MagicMock(return_value=1)
    r.hgetall = MagicMock(return_value={})
    r.delete = MagicMock(return_value=1)
    return r


def _task_hash(task_id: str, name: str = "Test task", gate: str = "s1") -> dict:
    return {
        b"id": task_id.encode(),
        b"name": name.encode(),
        b"gate": gate.encode(),
        b"assignees": json_module.dumps([]).encode(),
        b"start_date": b"",
        b"end_date": b"",
        b"depends_on": json_module.dumps([]).encode(),
        b"status": b"todo",
        b"progress": b"0",
    }


@pytest.mark.asyncio
async def test_create_task():
    r = _make_redis_mock()
    store = TaskStore(r)
    data = TaskCreate(name="Test task", gate="s1")
    task = await store.create("my-project", data)
    assert task.name == "Test task"
    assert task.gate == "s1"
    assert len(task.id) == 8
    assert r.hset.called


@pytest.mark.asyncio
async def test_list_tasks():
    r = _make_redis_mock()

    task_id = "abc12345"
    stored_hash = _task_hash(task_id)
    r.hgetall = MagicMock(return_value=stored_hash)

    async def fake_scan_iter(pattern):
        yield f"finefab:project:my-project:task:{task_id}"

    r.scan_iter = fake_scan_iter

    store = TaskStore(r)
    tasks = await store.list_tasks("my-project")
    assert len(tasks) == 1
    assert tasks[0].id == task_id
    assert tasks[0].gate == "s1"


@pytest.mark.asyncio
async def test_list_tasks_filter_by_gate():
    r = _make_redis_mock()

    task_id = "abc12345"
    stored_hash = _task_hash(task_id, gate="s2")
    r.hgetall = MagicMock(return_value=stored_hash)

    async def fake_scan_iter(pattern):
        yield f"finefab:project:my-project:task:{task_id}"

    r.scan_iter = fake_scan_iter

    store = TaskStore(r)
    # Filter for s1 — should be empty since task is s2
    tasks = await store.list_tasks("my-project", gate="s1")
    assert tasks == []

    # Filter for s2 — should return the task
    tasks = await store.list_tasks("my-project", gate="s2")
    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_update_task():
    r = _make_redis_mock()
    task_id = "abc12345"
    r.hgetall = MagicMock(return_value=_task_hash(task_id))
    store = TaskStore(r)
    updated = await store.update("my-project", task_id, {"status": "in_progress", "progress": 50})
    assert updated.status == "in_progress"
    assert updated.progress == 50
    assert r.hset.called


@pytest.mark.asyncio
async def test_delete_task():
    r = _make_redis_mock()
    r.delete = MagicMock(return_value=1)
    store = TaskStore(r)
    result = await store.delete("my-project", "abc12345")
    assert result is True
    r.delete.assert_called_once_with("finefab:project:my-project:task:abc12345")
