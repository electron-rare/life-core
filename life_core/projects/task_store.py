"""Redis CRUD store for project tasks."""
from __future__ import annotations

import json as json_module
import logging
import uuid
from typing import Any

from life_core.projects.models import Task, TaskCreate

logger = logging.getLogger("life_core.projects.task_store")


def _task_key(project: str, task_id: str) -> str:
    return f"finefab:project:{project}:task:{task_id}"


def _task_prefix(project: str) -> str:
    return f"finefab:project:{project}:task:"


class TaskStore:
    """CRUD operations for tasks stored as Redis hashes."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def create(self, project: str, data: TaskCreate) -> Task:
        task_id = uuid.uuid4().hex[:8]
        task = Task(id=task_id, **data.model_dump())
        key = _task_key(project, task_id)
        self._redis.hset(key, mapping={
            k: json_module.dumps(v) if isinstance(v, (list, dict)) else (v if v is not None else "")
            for k, v in task.model_dump().items()
        })
        return task

    async def list_tasks(self, project: str, gate: str | None = None) -> list[Task]:
        tasks: list[Task] = []
        prefix = _task_prefix(project)
        async for key in self._redis.scan_iter(f"{prefix}*"):
            try:
                raw = self._redis.hgetall(key)
                if not raw:
                    continue
                decoded = _decode_hash(raw)
                task = Task(**decoded)
                if gate is None or task.gate == gate:
                    tasks.append(task)
            except Exception as e:
                logger.warning("Failed to decode task at %s: %s", key, e)
        return tasks

    async def update(self, project: str, task_id: str, updates: dict) -> Task:
        key = _task_key(project, task_id)
        raw = self._redis.hgetall(key)
        if not raw:
            raise KeyError(f"Task {task_id} not found in project {project}")
        current = _decode_hash(raw)
        current.update({k: v for k, v in updates.items() if v is not None})
        task = Task(**current)
        self._redis.hset(key, mapping={
            k: json_module.dumps(v) if isinstance(v, (list, dict)) else (v if v is not None else "")
            for k, v in task.model_dump().items()
        })
        return task

    async def delete(self, project: str, task_id: str) -> bool:
        key = _task_key(project, task_id)
        result = self._redis.delete(key)
        return bool(result)


def _decode_hash(raw: dict) -> dict:
    """Decode a Redis hash (bytes or str values) into Python types."""
    out: dict = {}
    for k, v in raw.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        # Try JSON decode for lists/dicts/numbers
        try:
            out[key] = json_module.loads(val)
        except (json_module.JSONDecodeError, TypeError):
            out[key] = val if val != "" else None
    return out
