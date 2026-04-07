"""Projects gates dashboard API.

Reads .kill-life.yaml from GitHub repos via the GitHub API.
Results are cached in Redis (TTL 5 min) to avoid rate limits.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("life_core.projects")

router = APIRouter(prefix="/projects", tags=["Projects"])

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
            return json.loads(data)
    except Exception as e:
        logger.warning("Cache get failed for %s: %s", key, e)
    return None


async def _cache_set(key: str, value: Any) -> None:
    if _redis is None:
        return
    try:
        _redis.set(key, json.dumps(value), ex=_CACHE_TTL)
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
