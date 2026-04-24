"""Router tests for ``life_core.evaluations.router`` (T1.12a)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.evaluations.router import get_session, router


def _make_app(session_factory) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = session_factory
    return app


def test_run_evaluation_happy_path():
    session = MagicMock()
    app = _make_app(lambda: session)
    client = TestClient(app)

    fake_eval = MagicMock()
    fake_eval.id = uuid4()
    fake_eval.comparator = "spec_coverage"
    fake_eval.score = 0.9
    fake_eval.details = {"method": "llm_as_judge"}

    async def _fake_runner(_s, _a, _b):
        return [fake_eval]

    with patch(
        "life_core.evaluations.router.run_evaluation", side_effect=_fake_runner
    ):
        r = client.post(
            f"/evaluations/run?llm_agent_run_id={uuid4()}&"
            f"human_agent_run_id={uuid4()}"
        )

    assert r.status_code == 200
    body = r.json()
    assert body["evaluations"][0]["comparator"] == "spec_coverage"
    assert body["evaluations"][0]["score"] == 0.9
    assert body["evaluations"][0]["details"] == {"method": "llm_as_judge"}
    assert body["evaluations"][0]["id"] == str(fake_eval.id)


def test_run_evaluation_missing_query_returns_422():
    session = MagicMock()
    app = _make_app(lambda: session)
    client = TestClient(app)

    r = client.post("/evaluations/run")
    assert r.status_code == 422


def test_run_evaluation_bad_uuid_returns_422():
    session = MagicMock()
    app = _make_app(lambda: session)
    client = TestClient(app)

    r = client.post(
        "/evaluations/run?llm_agent_run_id=not-a-uuid&"
        f"human_agent_run_id={uuid4()}"
    )
    assert r.status_code == 422


def test_run_evaluation_value_error_becomes_422():
    session = MagicMock()
    app = _make_app(lambda: session)
    client = TestClient(app)

    async def _raise(_s, _a, _b):
        raise ValueError("agent_run(s) not found")

    with patch(
        "life_core.evaluations.router.run_evaluation", side_effect=_raise
    ):
        r = client.post(
            f"/evaluations/run?llm_agent_run_id={uuid4()}&"
            f"human_agent_run_id={uuid4()}"
        )

    assert r.status_code == 422
    assert "agent_run" in r.json()["detail"]


def test_run_evaluation_multiple_results():
    session = MagicMock()
    app = _make_app(lambda: session)
    client = TestClient(app)

    ev1 = MagicMock(comparator="hardware_diff", score=0.7, details={"x": 1})
    ev1.id = uuid4()
    ev2 = MagicMock(comparator="spec_coverage", score=0.9, details={"y": 2})
    ev2.id = uuid4()

    async def _fake_runner(_s, _a, _b):
        return [ev1, ev2]

    with patch(
        "life_core.evaluations.router.run_evaluation", side_effect=_fake_runner
    ):
        r = client.post(
            f"/evaluations/run?llm_agent_run_id={uuid4()}&"
            f"human_agent_run_id={uuid4()}"
        )

    assert r.status_code == 200
    body = r.json()
    assert len(body["evaluations"]) == 2
    comparators = {e["comparator"] for e in body["evaluations"]}
    assert comparators == {"hardware_diff", "spec_coverage"}
