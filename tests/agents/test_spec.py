"""Tests for the spec agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from life_core.agents.spec import SpecAgent


@pytest.fixture
def intake():
    fixture_path = (
        Path(__file__).parent.parent / "fixtures" / "intakes" / "hw-simple.json"
    )
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_spec_agent_generates_rfc2119_section_headers(intake, monkeypatch):
    async def fake_llm(prompt: str) -> str:
        return (
            "# 01_spec\n\n## Requirements\n\nThe system MUST manage 16 cells.\n"
            "The system SHOULD expose INA226 readings.\n"
        )

    monkeypatch.setattr("life_core.agents.spec.call_llm", fake_llm)

    agent = SpecAgent()
    result = await agent.run({"intake": intake, "compliance_profile": "prototype"})

    assert result.ok is True
    assert "## Requirements" in result.output
    assert "MUST" in result.output


@pytest.mark.asyncio
async def test_spec_agent_fails_when_llm_skips_requirements(intake, monkeypatch):
    async def fake_llm(prompt: str) -> str:
        return "# no requirements here\n\nplain text only"

    monkeypatch.setattr("life_core.agents.spec.call_llm", fake_llm)

    agent = SpecAgent()
    result = await agent.run({"intake": intake})

    assert result.ok is False
    assert "missing RFC2119 markers" in result.reasons
