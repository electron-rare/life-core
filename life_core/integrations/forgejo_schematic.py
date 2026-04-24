"""V1.7 Track II Task 10 — list Forgejo repos hosting a KiCad project.

Calls `git.saillant.cc/api/v1/orgs/factory-4-life/repos`, then probes
the root directory of each repo for a `.kicad_pro` file. Returns
``{projects: [{repo, last_commit, kicad_pcb_url}]}`` with a 60 s cache.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

CACHE_TTL_S = 60.0
TIMEOUT_S = 5.0
ORG = "factory-4-life"


_cache: dict[str, Any] | None = None
_cache_time: float = 0.0
_lock = asyncio.Lock()


def _cache_clear_for_test() -> None:
    """Reset the in-module cache; called from tests only."""
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0


def _forgejo_url() -> str:
    return os.environ.get(
        "FORGEJO_URL", "https://git.saillant.cc"
    ).rstrip("/")


def _headers() -> dict[str, str]:
    token = os.environ.get("FORGEJO_TOKEN", "")
    if token:
        return {"Authorization": f"token {token}"}
    return {}


async def _list_repos(
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    resp = await client.get(
        f"{_forgejo_url()}/api/v1/orgs/{ORG}/repos",
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


async def _probe_contents(
    client: httpx.AsyncClient, repo: str
) -> list[dict[str, Any]]:
    resp = await client.get(
        f"{_forgejo_url()}/api/v1/repos/{ORG}/{repo}/contents/",
        headers=_headers(),
    )
    if resp.status_code != 200:
        return []
    return resp.json()


async def list_kicad_projects() -> dict[str, Any]:
    """Return all factory-4-life repos whose root has a .kicad_pro file."""
    global _cache, _cache_time
    async with _lock:
        now = time.monotonic()
        if _cache is not None and (now - _cache_time) < CACHE_TTL_S:
            return _cache

        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            repos = await _list_repos(client)
            projects: list[dict[str, Any]] = []
            for repo in repos:
                contents = await _probe_contents(client, repo["name"])
                pro = next(
                    (
                        f
                        for f in contents
                        if f.get("type") == "file"
                        and f.get("name", "").endswith(".kicad_pro")
                    ),
                    None,
                )
                if not pro:
                    continue
                pcb = next(
                    (
                        f["name"]
                        for f in contents
                        if f.get("type") == "file"
                        and f.get("name", "").endswith(".kicad_pcb")
                    ),
                    None,
                )
                branch = repo.get("default_branch", "main")
                pcb_url = (
                    f"{_forgejo_url()}/{ORG}/{repo['name']}"
                    f"/raw/branch/{branch}/{pcb}"
                    if pcb
                    else None
                )
                projects.append(
                    {
                        "repo": repo["name"],
                        "last_commit": repo.get("updated_at"),
                        "kicad_pcb_url": pcb_url,
                    }
                )

        _cache = {"projects": projects}
        _cache_time = now
        return _cache
