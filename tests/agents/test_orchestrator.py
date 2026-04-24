"""Orchestrator tests (T1.8c)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from life_core.agents.contract import AgentRequest
from life_core.agents.orchestrator import run_agent
from life_core.generators.base import GenerationOutcome


@pytest.mark.asyncio
async def test_run_agent_spec_returns_pass_on_generator_ok(tmp_path):
    req = AgentRequest(
        deliverable_slug="s-spec",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
        upstream_artifacts=[],
        context={"brief": "b", "constraints": []},
        hitl_mode="sync",
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationOutcome(
        ok=True,
        data=(
            b"---\n"
            b"description: x\ninputs: []\noutputs: []\nconstraints: []\n"
            b"acceptance_criteria: []\ncompliance: prototype\n"
            b"---\n#"
        ),
        errors=[],
        attempts=1,
    )
    session = MagicMock()
    ref = MagicMock(
        id="uuid",
        storage_path=tmp_path / "a.md",
        version=1,
        type="spec",
        deliverable_slug="s-spec",
    )
    with (
        patch(
            "life_core.agents.orchestrator._pick_generator",
            return_value=fake_gen,
        ),
        patch("life_core.agents.orchestrator.write", return_value=ref),
        patch(
            "life_core.agents.orchestrator._wait_human_decision",
            AsyncMock(return_value="approve"),
        ),
    ):
        env = await run_agent("spec", req, session=session, volume_root=tmp_path)
    assert env.result.ok is True
    assert env.result.verdict == "GateSpecPass"
    assert env.result.inner_state_final == "APPROVED"
    assert env.job_id is not None


@pytest.mark.asyncio
async def test_run_agent_fail_on_generator_ko():
    req = AgentRequest(
        deliverable_slug="s",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
        upstream_artifacts=[],
        context={},
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationOutcome(
        ok=False, data=b"", errors=["missing frontmatter"], attempts=3
    )
    session = MagicMock()
    with patch(
        "life_core.agents.orchestrator._pick_generator", return_value=fake_gen
    ):
        env = await run_agent("spec", req, session=session, volume_root=None)
    assert env.result.ok is False
    assert env.result.verdict == "GateSpecFail"
    assert "missing frontmatter" in env.result.reasons
    assert env.result.inner_state_final == "REJECTED"


@pytest.mark.asyncio
async def test_run_agent_async_mode_returns_without_waiting(tmp_path):
    req = AgentRequest(
        deliverable_slug="s-async",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
        upstream_artifacts=[],
        context={},
        hitl_mode="async",
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationOutcome(
        ok=True, data=b"ok", errors=[], attempts=1
    )
    session = MagicMock()
    ref = MagicMock(storage_path=tmp_path / "b.md", version=2)
    wait_mock = AsyncMock(return_value="approve")
    with (
        patch(
            "life_core.agents.orchestrator._pick_generator",
            return_value=fake_gen,
        ),
        patch("life_core.agents.orchestrator.write", return_value=ref),
        patch(
            "life_core.agents.orchestrator._wait_human_decision",
            wait_mock,
        ),
    ):
        env = await run_agent(
            "spec", req, session=session, volume_root=tmp_path
        )
    # async mode: we do NOT await the human decision
    wait_mock.assert_not_called()
    assert env.result.ok is True
    # async mode leaves inner_state_final unset (still REVIEW in DB)
    assert env.result.inner_state_final is None
    assert env.result.artifacts[0]["version"] == 2


@pytest.mark.asyncio
async def test_run_agent_sync_timeout_yields_reject(tmp_path):
    req = AgentRequest(
        deliverable_slug="s-timeout",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
        hitl_mode="sync",
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationOutcome(
        ok=True, data=b"ok", errors=[], attempts=1
    )
    session = MagicMock()
    ref = MagicMock(storage_path=tmp_path / "c.md", version=1)
    with (
        patch(
            "life_core.agents.orchestrator._pick_generator",
            return_value=fake_gen,
        ),
        patch("life_core.agents.orchestrator.write", return_value=ref),
        patch(
            "life_core.agents.orchestrator._wait_human_decision",
            AsyncMock(return_value="timeout"),
        ),
    ):
        env = await run_agent(
            "spec", req, session=session, volume_root=tmp_path
        )
    assert env.result.ok is False
    assert env.result.verdict == "GateSpecFail"
    assert env.result.inner_state_final == "TIMEOUT"


def test_pick_generator_routes_deliverable_types():
    from life_core.agents.orchestrator import _pick_generator
    from life_core.generators.firmware_generator import FirmwareGenerator
    from life_core.generators.kicad_generator import KicadGenerator
    from life_core.generators.spec_generator import SpecGenerator
    from life_core.generators.spice_generator import SpiceGenerator

    assert isinstance(_pick_generator("spec", "spec"), SpecGenerator)
    assert isinstance(_pick_generator("impl", "hardware"), KicadGenerator)
    assert isinstance(_pick_generator("impl", "firmware"), FirmwareGenerator)
    assert isinstance(_pick_generator("impl", "simulation"), SpiceGenerator)
    # unknown deliverable_type falls back to SpecGenerator
    assert isinstance(_pick_generator("impl", "bom"), SpecGenerator)


@pytest.mark.asyncio
async def test_wait_human_decision_detects_approved(monkeypatch):
    from life_core.agents import orchestrator as orch

    # Collapse poll interval and review timeout so the test finishes quickly.
    monkeypatch.setattr(orch, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(orch, "_REVIEW_TIMEOUT_S", 1)

    session = MagicMock()
    row = MagicMock()
    row.inner_state = "APPROVED"
    session.get.return_value = row
    from uuid import uuid4

    decision = await orch._wait_human_decision(uuid4(), session)
    assert decision == "approve"


@pytest.mark.asyncio
async def test_wait_human_decision_detects_rejected(monkeypatch):
    from life_core.agents import orchestrator as orch

    monkeypatch.setattr(orch, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(orch, "_REVIEW_TIMEOUT_S", 1)

    session = MagicMock()
    row = MagicMock()
    row.inner_state = "REJECTED"
    session.get.return_value = row
    from uuid import uuid4

    decision = await orch._wait_human_decision(uuid4(), session)
    assert decision == "reject"


@pytest.mark.asyncio
async def test_wait_human_decision_times_out(monkeypatch):
    from life_core.agents import orchestrator as orch

    # Stay in REVIEW the whole deadline; expect a timeout verdict.
    monkeypatch.setattr(orch, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(orch, "_REVIEW_TIMEOUT_S", 0)

    session = MagicMock()
    row = MagicMock()
    row.inner_state = "REVIEW"
    session.get.return_value = row
    from uuid import uuid4

    decision = await orch._wait_human_decision(uuid4(), session)
    assert decision == "timeout"


@pytest.mark.asyncio
async def test_run_agent_impl_role_produces_impl_verdicts(tmp_path):
    req = AgentRequest(
        deliverable_slug="s-impl",
        deliverable_type="firmware",
        outer_state="impl",
        compliance_profile="prototype",
        hitl_mode="sync",
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationOutcome(
        ok=False, data=b"", errors=["bad zephyr layout"], attempts=1
    )
    session = MagicMock()
    with patch(
        "life_core.agents.orchestrator._pick_generator", return_value=fake_gen
    ):
        env = await run_agent(
            "impl", req, session=session, volume_root=tmp_path
        )
    assert env.result.ok is False
    assert env.result.verdict == "GateImplFail"
