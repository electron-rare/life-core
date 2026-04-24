"""V1.7 Track II — /governance endpoint.

Aggregates branch protection + open PR count for F4L repos on
both GitHub (legacy mirror) and Forgejo (new source of truth,
see Plan 1). Cached 60 s.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from life_core.events.broker import get_broker
from life_core.events.schema import Event, EventType

CACHE_TTL_S = 60.0
TIMEOUT_S = 5.0
ORG = "factory-4-life"

F4L_REPOS = (
    "life-core",
    "life-web",
    "life-reborn",
    "finefab-shared",
    "makelife-cad",
    "makelife-firmware",
    "makelife-hard",
    "KIKI-models-tuning",
    "finefab-life",
    "rag-web",
)


_cache: dict[str, Any] | None = None
_cache_time: float = 0.0
_lock = asyncio.Lock()


def _cache_clear_for_test() -> None:
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0


def _forgejo_url() -> str:
    return os.environ.get(
        "FORGEJO_URL", "https://git.saillant.cc"
    ).rstrip("/")


def _forgejo_headers() -> dict[str, str]:
    tok = os.environ.get("FORGEJO_TOKEN", "")
    return {"Authorization": f"token {tok}"} if tok else {}


async def _forgejo_snapshot() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.get(
            f"{_forgejo_url()}/api/v1/orgs/{ORG}/repos",
            headers=_forgejo_headers(),
        )
        resp.raise_for_status()
        repos = resp.json()
        out: list[dict[str, Any]] = []
        for r in repos:
            name = r["name"]
            branch = r.get("default_branch", "main")
            bp_resp = await client.get(
                f"{_forgejo_url()}/api/v1/repos/{ORG}/{name}/"
                f"branch_protections/{branch}",
                headers=_forgejo_headers(),
            )
            protected = False
            if bp_resp.status_code == 200:
                body = bp_resp.json()
                protected = int(body.get("required_approvals", 0)) >= 1
            prs_resp = await client.get(
                f"{_forgejo_url()}/api/v1/repos/{ORG}/{name}/"
                "pulls?state=open",
                headers=_forgejo_headers(),
            )
            open_prs = (
                len(prs_resp.json())
                if prs_resp.status_code == 200
                else 0
            )
            out.append(
                {
                    "host": "forgejo",
                    "repo": name,
                    "branch_protected": protected,
                    "open_prs": open_prs,
                }
            )
        return out


async def _github_snapshot(repos: tuple[str, ...]) -> list[dict[str, Any]]:
    tok = os.environ.get("GITHUB_TOKEN", "") or os.environ.get(
        "KILL_LIFE_GITHUB_TOKEN", ""
    )
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        for repo in repos:
            protected = False
            open_prs = 0
            try:
                r = await client.get(
                    f"https://api.github.com/repos/electron-rare/"
                    f"{repo}/branches/main/protection",
                    headers=headers,
                )
                protected = r.status_code == 200
                prs = await client.get(
                    f"https://api.github.com/repos/electron-rare/"
                    f"{repo}/pulls?state=open",
                    headers=headers,
                )
                if prs.status_code == 200:
                    open_prs = len(prs.json())
            except Exception:
                pass
            out.append(
                {
                    "host": "github",
                    "repo": repo,
                    "branch_protected": protected,
                    "open_prs": open_prs,
                }
            )
    return out


async def _collect() -> dict[str, Any]:
    fj, gh = await asyncio.gather(
        _forgejo_snapshot(),
        _github_snapshot(F4L_REPOS),
    )
    return {
        "repos": [*fj, *gh],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_governance() -> dict[str, Any]:
    global _cache, _cache_time
    async with _lock:
        now = time.monotonic()
        if _cache is None or (now - _cache_time) > CACHE_TTL_S:
            _cache = await _collect()
            _cache_time = now
            await _emit(_cache)
        return _cache


async def _emit(payload: dict[str, Any]) -> None:
    broker = get_broker()
    unprotected = [
        r["repo"]
        for r in payload["repos"]
        if r["host"] == "forgejo" and not r["branch_protected"]
    ]
    await broker.publish(
        Event(
            type=EventType.INFRA_CONTAINER,
            data={
                "kind": "governance_scan",
                "unprotected_count": len(unprotected),
                "unprotected": unprotected[:20],
            },
            timestamp=datetime.now(timezone.utc),
        )
    )
