"""V1.7 Track II Task 4 — assert legacy polling routes are removed.

The /stats and /goose/stats handlers have been deleted in favor of
the unified SSE /events stream (see life_core/api.py#/events).
This regression guard makes sure they cannot be silently
re-introduced.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_stats_route_is_404():
    from life_core.api import app

    client = TestClient(app)
    resp = client.get("/stats")
    assert resp.status_code == 404


def test_goose_stats_route_is_404():
    from life_core.api import app

    client = TestClient(app)
    # historical path was /goose-stats; current impl mounts goose
    # router under /goose prefix so the real legacy endpoint is
    # /goose/stats. Both must be absent post-Task 4.
    assert client.get("/goose-stats").status_code == 404
    assert client.get("/goose/stats").status_code == 404
