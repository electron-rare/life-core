"""Git sync helpers — fetch/push project YAML from/to GitHub registry."""
from __future__ import annotations

import base64
import json as json_module
import logging
import os

import httpx
import yaml

logger = logging.getLogger("life_core.projects.git_sync")

_REGISTRY_REPO = "L-electron-Rare/life-project"
_REGISTRY_PATH = "projects"
_GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def fetch_remote_yaml(name: str) -> tuple[str, str]:
    """Fetch .kill-life.yaml from the registry repo.

    Returns (content_str, sha).
    Raises httpx.HTTPStatusError on non-2xx.
    """
    url = f"{_GITHUB_API}/repos/{_REGISTRY_REPO}/contents/{_REGISTRY_PATH}/{name}.yaml"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    content_b64 = data.get("content", "")
    # GitHub base64 includes newlines
    content = base64.b64decode(content_b64.replace("\n", "")).decode()
    sha = data.get("sha", "")
    return content, sha


async def push_yaml(name: str, content: str, sha: str, message: str) -> str:
    """PUT project YAML to the registry repo.

    Returns the commit SHA of the new commit.
    """
    url = f"{_GITHUB_API}/repos/{_REGISTRY_REPO}/contents/{_REGISTRY_PATH}/{name}.yaml"
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "sha": sha,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.put(url, headers=_headers(), content=json_module.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
    return data.get("commit", {}).get("sha", "")


def project_to_yaml(project_dict: dict) -> str:
    """Convert a project dict to a kill-life YAML string."""
    wrapped = {"kill_life": project_dict}
    return yaml.dump(wrapped, default_flow_style=False, allow_unicode=True)
