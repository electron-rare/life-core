"""V1.7 Track II Task 11 — /governance endpoint (GitHub + Forgejo)."""
from __future__ import annotations

import asyncio

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")
    monkeypatch.setenv(
        "FORGEJO_URL", "https://git.saillant.cc"
    )
    monkeypatch.setenv("FORGEJO_TOKEN", "fjt-abc")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-abc")


@respx.mock
def test_governance_aggregates(monkeypatch):
    from life_core.api import app
    from life_core.integrations import governance as gov

    gov._cache_clear_for_test()

    async def fake_github(repos):
        return [
            {
                "host": "github",
                "repo": r,
                "branch_protected": True,
                "open_prs": 2,
            }
            for r in repos
        ]

    monkeypatch.setattr(gov, "_github_snapshot", fake_github)

    respx.get(
        "https://git.saillant.cc/api/v1/orgs/factory-4-life/repos"
    ).mock(
        return_value=Response(
            200,
            json=[
                {
                    "name": "life-core",
                    "default_branch": "main",
                },
                {
                    "name": "life-web",
                    "default_branch": "main",
                },
            ],
        )
    )
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "life-core/branch_protections/main"
    ).mock(
        return_value=Response(200, json={"required_approvals": 1})
    )
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "life-web/branch_protections/main"
    ).mock(
        return_value=Response(200, json={"required_approvals": 0})
    )
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "life-core/pulls?state=open"
    ).mock(return_value=Response(200, json=[{"id": 1}]))
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "life-web/pulls?state=open"
    ).mock(return_value=Response(200, json=[]))

    client = TestClient(app)
    resp = client.get(
        "/governance",
        headers={"Authorization": "Bearer sekret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "repos" in body
    hosts = {r["host"] for r in body["repos"]}
    assert hosts == {"github", "forgejo"}
    fj = [r for r in body["repos"] if r["host"] == "forgejo"]
    assert any(
        r["repo"] == "life-core" and r["branch_protected"] and r["open_prs"] == 1
        for r in fj
    )


def test_governance_is_cached_60s(monkeypatch):
    from life_core.integrations import governance as gov

    call_count = 0

    async def fake_collect():
        nonlocal call_count
        call_count += 1
        return {"repos": []}

    monkeypatch.setattr(gov, "_collect", fake_collect)
    gov._cache_clear_for_test()

    async def run():
        for _ in range(5):
            await gov.get_governance()

    asyncio.run(run())
    assert call_count == 1
