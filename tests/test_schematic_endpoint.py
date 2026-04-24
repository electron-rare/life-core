"""V1.7 Track II Task 10 — /schematic endpoint via Forgejo."""
from __future__ import annotations

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


@respx.mock
def test_schematic_lists_kicad_projects():
    from life_core.api import app
    from life_core.integrations import forgejo_schematic as fs

    fs._cache_clear_for_test()

    respx.get(
        "https://git.saillant.cc/api/v1/orgs/factory-4-life/repos"
    ).mock(
        return_value=Response(
            200,
            json=[
                {
                    "name": "makelife-hard",
                    "updated_at": "2026-04-20T10:00:00Z",
                    "default_branch": "main",
                },
                {
                    "name": "life-core",
                    "updated_at": "2026-04-22T09:00:00Z",
                    "default_branch": "main",
                },
                {
                    "name": "another-board",
                    "updated_at": "2026-04-21T12:00:00Z",
                    "default_branch": "main",
                },
            ],
        )
    )
    # makelife-hard has .kicad_pro + .kicad_pcb
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "makelife-hard/contents/"
    ).mock(
        return_value=Response(
            200,
            json=[
                {"name": "board.kicad_pro", "type": "file"},
                {"name": "board.kicad_pcb", "type": "file"},
            ],
        )
    )
    # life-core has no KiCad files
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "life-core/contents/"
    ).mock(
        return_value=Response(
            200, json=[{"name": "README.md", "type": "file"}]
        )
    )
    # another-board has .kicad_pro only
    respx.get(
        "https://git.saillant.cc/api/v1/repos/factory-4-life/"
        "another-board/contents/"
    ).mock(
        return_value=Response(
            200,
            json=[
                {"name": "main.kicad_pro", "type": "file"},
                {"name": "main.kicad_pcb", "type": "file"},
            ],
        )
    )

    client = TestClient(app)
    resp = client.get(
        "/schematic", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["projects"]) == 2
    names = {p["repo"] for p in body["projects"]}
    assert names == {"makelife-hard", "another-board"}
    by_name = {p["repo"]: p for p in body["projects"]}
    assert (
        by_name["makelife-hard"]["last_commit"]
        == "2026-04-20T10:00:00Z"
    )
    assert by_name["makelife-hard"]["kicad_pcb_url"].endswith(
        "board.kicad_pcb"
    )
