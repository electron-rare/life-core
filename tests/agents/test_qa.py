"""Tests for the QA agent."""

from __future__ import annotations

import pytest

from life_core.agents.qa import QaAgent


@pytest.mark.asyncio
async def test_qa_pass_when_all_checks_ok(monkeypatch):
    async def fake_llm(prompt: str) -> str:
        return '{"verdict":"pass","reasons":[]}'

    monkeypatch.setattr("life_core.agents.qa.call_llm", fake_llm)

    agent = QaAgent()
    result = await agent.run(
        {
            "deliverable_id": "kxkm-v1",
            "gate": "G-impl",
            "artefacts": {"erc_errors": 0, "drc_errors": 0, "build_ok": True},
            "compliance_profile": "prototype",
        }
    )
    assert result.ok is True
    assert result.output == "pass"
    assert result.reasons == []


@pytest.mark.asyncio
async def test_qa_fail_with_compliance_category(monkeypatch):
    async def fake_llm(prompt: str) -> str:
        return (
            '{"verdict":"fail","reasons":["EMC Class B missing"],'
            '"category":"compliance"}'
        )

    monkeypatch.setattr("life_core.agents.qa.call_llm", fake_llm)

    agent = QaAgent()
    result = await agent.run(
        {
            "deliverable_id": "kxkm-v1",
            "gate": "G-impl",
            "artefacts": {"erc_errors": 0, "drc_errors": 2},
            "compliance_profile": "iot_wifi_eu",
        }
    )
    assert result.ok is False
    assert result.output == "fail"
    assert any("EMC" in r for r in result.reasons)
    assert any("category:compliance" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_qa_fails_on_invalid_llm_json(monkeypatch):
    async def fake_llm(prompt: str) -> str:
        return "not valid json"

    monkeypatch.setattr("life_core.agents.qa.call_llm", fake_llm)

    agent = QaAgent()
    result = await agent.run(
        {"deliverable_id": "x", "gate": "G-impl", "artefacts": {}}
    )
    assert result.ok is False
    assert result.output == "fail"
    assert any("llm error" in r for r in result.reasons)
