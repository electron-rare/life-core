"""Tests for AgentBase scaffold."""

from __future__ import annotations

import pytest

from life_core.agents.base import AgentBase, AgentResult


class DummyAgent(AgentBase):
    role = "dummy"

    async def run(self, payload):
        return AgentResult(ok=True, output="hello", reasons=[])


@pytest.mark.asyncio
async def test_dummy_agent_runs():
    agent = DummyAgent()
    result = await agent.run({"foo": "bar"})
    assert result.ok is True
    assert result.output == "hello"
    assert result.reasons == []


def test_agent_base_requires_role():
    class Broken(AgentBase):
        pass

    with pytest.raises(NotImplementedError):
        Broken()
