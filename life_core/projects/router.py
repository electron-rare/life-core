"""Projects gates dashboard API.

Reads .kill-life.yaml from GitHub repos via the GitHub API.
Results are cached in Redis (TTL 5 min) to avoid rate limits.
"""

from __future__ import annotations

import json as json_module
import logging
import os
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, HTTPException

from life_core.projects.models import (
    Gate,
    ProjectCreate,
    ProjectUpdate,
    TaskCreate,
)
from life_core.projects.task_store import TaskStore
from life_core.projects.git_sync import fetch_remote_yaml, project_to_yaml, push_yaml
from life_core.projects.team import get_team_members

logger = logging.getLogger("life_core.projects")

router = APIRouter(prefix="/projects", tags=["Projects"])
team_router = APIRouter(prefix="/team", tags=["Team"])

# Redis cache TTL (seconds)
_CACHE_TTL = 300

# Registry repo and path
_REGISTRY_REPO = "L-electron-Rare/life-project"
_REGISTRY_PATH = "projects"

# Module-level Redis client (injected from api.py lifespan)
_redis = None


def set_redis(redis_client: Any) -> None:
    """Inject Redis client for caching."""
    global _redis
    _redis = redis_client


def _github_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _cache_get(key: str) -> Any | None:
    if _redis is None:
        return None
    try:
        data = _redis.get(key)
        if data:
            return json_module.loads(data)
    except Exception as e:
        logger.warning("Cache get failed for %s: %s", key, e)
    return None


async def _cache_set(key: str, value: Any) -> None:
    if _redis is None:
        return
    try:
        _redis.set(key, json_module.dumps(value), ex=_CACHE_TTL)
    except Exception as e:
        logger.warning("Cache set failed for %s: %s", key, e)


def _parse_gate_status(gates: dict) -> dict[str, dict]:
    """Normalize gate entries to consistent shape."""
    result: dict[str, dict] = {}
    for gate_name in ("s0", "s1", "s2", "s3"):
        raw = gates.get(gate_name, {})
        if isinstance(raw, dict):
            result[gate_name] = {
                "status": raw.get("status", "pending"),
                "date": raw.get("date"),
            }
        else:
            result[gate_name] = {"status": "pending", "date": None}
    return result


def _parse_kill_life_yaml(content: str, filename: str) -> dict | None:
    """Parse a .kill-life.yaml content string. Returns None on parse error."""
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return None
        kl = data.get("kill_life", data)
        if not isinstance(kl, dict):
            return None

        project_name = kl.get("project") or filename.replace(".yaml", "")
        gates_raw = kl.get("gates", {})

        return {
            "name": project_name,
            "repo": kl.get("repo"),
            "client": kl.get("client"),
            "gates": _parse_gate_status(gates_raw if isinstance(gates_raw, dict) else {}),
            "hardware": kl.get("hardware"),
            "firmware": kl.get("firmware"),
        }
    except Exception as e:
        logger.warning("Failed to parse kill-life YAML %s: %s", filename, e)
        return None


async def _fetch_projects_from_github() -> list[dict]:
    """Fetch all project YAML files from the registry repo."""
    base_url = "https://api.github.com"
    headers = _github_headers()

    async with httpx.AsyncClient(timeout=15.0) as client:
        # List directory contents
        resp = await client.get(
            f"{base_url}/repos/{_REGISTRY_REPO}/contents/{_REGISTRY_PATH}",
            headers=headers,
        )
        if resp.status_code == 404:
            logger.warning("Registry repo %s path %s not found", _REGISTRY_REPO, _REGISTRY_PATH)
            return []
        if resp.status_code == 403:
            logger.warning("GitHub API rate limit hit or auth required")
            return []
        resp.raise_for_status()

        entries = resp.json()
        if not isinstance(entries, list):
            return []

        yaml_files = [e for e in entries if isinstance(e, dict) and e.get("name", "").endswith(".yaml")]

        projects: list[dict] = []
        for entry in yaml_files:
            download_url = entry.get("download_url")
            if not download_url:
                continue
            try:
                file_resp = await client.get(download_url, headers=headers)
                file_resp.raise_for_status()
                parsed = _parse_kill_life_yaml(file_resp.text, entry["name"])
                if parsed:
                    projects.append(parsed)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", entry.get("name"), e)

    return projects


async def _get_projects() -> list[dict]:
    """Return projects list, from cache if available."""
    cache_key = "projects:all"
    cached = await _cache_get(cache_key)
    if cached is not None:
        return cached

    projects = await _fetch_projects_from_github()
    await _cache_set(cache_key, projects)
    return projects


@router.get("")
async def list_projects():
    """List all tracked projects with their gate status."""
    try:
        projects = await _get_projects()
        return {"projects": projects, "count": len(projects)}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {e.response.status_code}") from e
    except Exception as e:
        logger.error("Failed to list projects: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch projects") from e


@router.get("/{project_name}")
async def get_project(project_name: str):
    """Get gate status for a single project by name."""
    try:
        projects = await _get_projects()
        for project in projects:
            if project.get("name") == project_name:
                return project
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project %s: %s", project_name, e)
        raise HTTPException(status_code=500, detail="Failed to fetch project") from e


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

def _project_key(name: str) -> str:
    return f"finefab:project:{name}:data"


def _redis_get_project(name: str) -> dict | None:
    if _redis is None:
        return None
    raw = _redis.get(_project_key(name))
    if raw:
        try:
            return json_module.loads(raw)
        except Exception:
            pass
    return None


def _redis_set_project(name: str, data: dict) -> None:
    if _redis is None:
        return
    _redis.set(_project_key(name), json_module.dumps(data))


@router.post("")
async def create_project(body: ProjectCreate):
    """Create a new project and store it in Redis."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    data = body.model_dump()
    _redis_set_project(body.name, data)
    return data


@router.put("/{name}")
async def update_project(name: str, body: ProjectUpdate):
    """Update an existing project in Redis."""
    existing = _redis_get_project(name)
    if existing is None:
        # Try GitHub cache as fallback
        try:
            content, _ = await fetch_remote_yaml(name)
            import yaml as _yaml
            raw = _yaml.safe_load(content)
            existing = raw.get("kill_life", raw) if isinstance(raw, dict) else {}
        except Exception:
            existing = {"name": name}
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "gates" in updates and updates["gates"]:
        updates["gates"] = {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in updates["gates"].items()}
    existing.update(updates)
    _redis_set_project(name, existing)
    return existing


@router.delete("/{name}")
async def delete_project(name: str):
    """Delete a project from Redis."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    deleted = _redis.delete(_project_key(name))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found")
    return {"deleted": name}


@router.post("/{name}/sync")
async def sync_project(name: str):
    """Fetch remote YAML and push local Redis copy to GitHub."""
    project = _redis_get_project(name)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found in Redis")
    try:
        _, sha = await fetch_remote_yaml(name)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            sha = ""
        else:
            raise HTTPException(status_code=502, detail=f"GitHub fetch failed: {e.response.status_code}") from e
    content = project_to_yaml(project)
    try:
        commit_sha = await push_yaml(name, content, sha, f"chore: sync {name} from life-core")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"GitHub push failed: {e.response.status_code}") from e
    return {"synced": name, "commit": commit_sha}


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------

@router.get("/{name}/tasks")
async def list_tasks(name: str, gate: str | None = None):
    """List tasks for a project, optionally filtered by gate."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    store = TaskStore(_redis)
    tasks = await store.list_tasks(name, gate=gate)
    return {"tasks": [t.model_dump() for t in tasks], "count": len(tasks)}


@router.post("/{name}/tasks")
async def create_task(name: str, body: TaskCreate):
    """Create a task for a project."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    store = TaskStore(_redis)
    task = await store.create(name, body)
    return task.model_dump()


@router.put("/{name}/tasks/{task_id}")
async def update_task(name: str, task_id: str, body: dict):
    """Update a task."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    store = TaskStore(_redis)
    try:
        task = await store.update(name, task_id, body)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return task.model_dump()


@router.delete("/{name}/tasks/{task_id}")
async def delete_task(name: str, task_id: str):
    """Delete a task."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    store = TaskStore(_redis)
    deleted = await store.delete(name, task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"deleted": task_id}


@router.get("/{name}/timeline")
async def get_timeline(name: str):
    """Return Gantt-format timeline data for a project."""
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    store = TaskStore(_redis)
    tasks = await store.list_tasks(name)
    items = [
        {
            "id": t.id,
            "name": t.name,
            "start": t.start_date,
            "end": t.end_date,
            "progress": t.progress,
            "dependencies": ",".join(t.depends_on),
            "custom_class": f"gate-{t.gate}",
        }
        for t in tasks
    ]
    return {"timeline": items}


# ---------------------------------------------------------------------------
# Team router
# ---------------------------------------------------------------------------

@team_router.get("/members")
async def list_team_members():
    """Return all team members (humans + AI agents)."""
    members = await get_team_members()
    return {"members": [m.model_dump() for m in members], "count": len(members)}
