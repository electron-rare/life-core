"""Integration tests for the /agents/{role}/run FastAPI route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from life_core.api import app

client = TestClient(app)


def test_agents_run_unknown_role_returns_404():
    r = client.post("/agents/unknown/run", json={})
    assert r.status_code == 404


def test_agents_run_spec_returns_job_id(monkeypatch):
    async def fake_llm(prompt):
        return "# x\n\n## Requirements\n\nMUST foo"

    monkeypatch.setattr("life_core.agents.spec.call_llm", fake_llm)
    r = client.post(
        "/agents/spec/run",
        json={
            "intake": {
                "title": "t",
                "normalized_payload": {"goal": "g", "constraints": []},
            },
            "compliance_profile": "prototype",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["result"]["ok"] is True


def test_agents_run_qa_returns_verdict(monkeypatch):
    async def fake_llm(prompt):
        return '{"verdict":"pass","reasons":[]}'

    monkeypatch.setattr("life_core.agents.qa.call_llm", fake_llm)
    r = client.post(
        "/agents/qa/run",
        json={
            "deliverable_id": "kxkm-v1",
            "gate": "G-impl",
            "artefacts": {"erc_errors": 0},
            "compliance_profile": "prototype",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["output"] == "pass"
