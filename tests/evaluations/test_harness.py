"""Tests for ``life_core.evaluations.harness`` (T1.9c)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from life_core.evaluations import harness


class _FakeArtifact:
    def __init__(self, type_: str, source: str, storage_path: str):
        self.id = uuid4()
        self.type = type_
        self.source = source
        self.storage_path = storage_path


class _FakeAgentRun:
    def __init__(self, slug: str):
        self.id = uuid4()
        self.deliverable_slug = slug


def _build_session(
    llm_run, human_run, llm_art, human_art
) -> MagicMock:
    """Assemble a MagicMock session wired to return the given rows."""

    session = MagicMock()

    def _get(_model, uid):
        return llm_run if uid == llm_run.id else human_run

    session.get.side_effect = _get

    # Successive scalars().first() calls → llm_art then human_art.
    first_results = iter([llm_art, human_art])

    def _execute(_query):
        result = MagicMock()
        result.scalars.return_value.first.return_value = next(first_results)
        return result

    session.execute.side_effect = _execute
    return session


def test_run_evaluation_missing_agent_runs_raises():
    session = MagicMock()
    session.get.return_value = None
    with pytest.raises(ValueError, match="agent_run"):
        asyncio.run(
            harness.run_evaluation(session, uuid4(), uuid4())
        )


def test_run_evaluation_missing_artifact_pair_raises():
    llm_run = _FakeAgentRun("slug-A")
    human_run = _FakeAgentRun("slug-A")

    session = MagicMock()
    session.get.side_effect = (
        lambda _m, uid: llm_run if uid == llm_run.id else human_run
    )
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    session.execute.return_value = result

    with pytest.raises(ValueError, match="artifact pair missing"):
        asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )


def test_run_evaluation_rejects_unknown_type():
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("bom", "llm", "/llm/p")  # bom ∉ _COMPARATORS
    human_art = _FakeArtifact("bom", "human", "/human/p")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    with pytest.raises(ValueError, match="Unknown type"):
        asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )


def test_run_evaluation_spec_happy_path():
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("spec", "llm", "/llm/spec.md")
    human_art = _FakeArtifact("spec", "human", "/human/spec.md")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    fake = {"score": 0.9, "details": {"method": "llm_as_judge"}}
    with patch(
        "life_core.evaluations.harness.spec_coverage_compare",
        return_value=fake,
    ), patch(
        "life_core.evaluations.harness.Path"
    ) as MockPath:
        MockPath.return_value.read_text.return_value = "text"
        evs = asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )

    assert len(evs) == 1
    ev = evs[0]
    assert ev.comparator == "spec_coverage"
    assert ev.score == 0.9
    assert ev.details == {"method": "llm_as_judge"}
    assert ev.llm_artifact_id == llm_art.id
    assert ev.human_artifact_id == human_art.id
    session.add.assert_called_once_with(ev)
    session.flush.assert_called_once()


def test_run_evaluation_hardware_dispatch():
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("hardware", "llm", "/llm/sch")
    human_art = _FakeArtifact("hardware", "human", "/human/sch")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    fake = {"score": 0.7, "details": {"bom_match": 2, "bom_total": 3}}
    with patch(
        "life_core.evaluations.harness.hardware_diff_compare",
        return_value=fake,
    ) as mock_hw:
        evs = asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )

    mock_hw.assert_called_once_with("/human/sch", "/llm/sch")
    assert evs[0].comparator == "hardware_diff"
    assert evs[0].score == 0.7


def test_run_evaluation_firmware_dispatch():
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("firmware", "llm", "/llm/pio")
    human_art = _FakeArtifact("firmware", "human", "/human/pio")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    fake = {"score": 0.5, "details": {}}
    with patch(
        "life_core.evaluations.harness.firmware_behavior_compare",
        return_value=fake,
    ) as mock_fw:
        evs = asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )

    mock_fw.assert_called_once_with("/human/pio", "/llm/pio")
    assert evs[0].comparator == "firmware_behavior"


def test_run_evaluation_simulation_dispatch():
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("simulation", "llm", "/llm/sim.cir")
    human_art = _FakeArtifact("simulation", "human", "/human/sim.cir")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    fake = {"score": 1.0, "details": {"rmse": 0.0}}
    with patch(
        "life_core.evaluations.harness.simulation_diff_compare",
        return_value=fake,
    ) as mock_sim:
        evs = asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )

    mock_sim.assert_called_once_with("/human/sim.cir", "/llm/sim.cir")
    assert evs[0].comparator == "simulation_diff"


def test_run_evaluation_compliance_uses_spec_coverage():
    """Compliance artifacts reuse the spec_coverage comparator."""
    llm_run = _FakeAgentRun("slug")
    human_run = _FakeAgentRun("slug")
    llm_art = _FakeArtifact("compliance", "llm", "/llm/c.md")
    human_art = _FakeArtifact("compliance", "human", "/human/c.md")
    session = _build_session(llm_run, human_run, llm_art, human_art)

    fake = {"score": 0.8, "details": {"method": "llm_as_judge"}}
    with patch(
        "life_core.evaluations.harness.spec_coverage_compare",
        return_value=fake,
    ), patch("life_core.evaluations.harness.Path") as MockPath:
        MockPath.return_value.read_text.return_value = "c"
        evs = asyncio.run(
            harness.run_evaluation(session, llm_run.id, human_run.id)
        )

    assert evs[0].comparator == "spec_coverage"
    assert evs[0].score == 0.8
