"""Contract tests for ``life_core.agents.contract`` (T1.8a)."""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from life_core.agents.contract import (
    AgentEnvelope,
    AgentRequest,
    AgentResult,
    ArtifactRef,
)


def test_agent_request_accepts_valid_payload():
    req = AgentRequest(
        deliverable_slug="s",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
        upstream_artifacts=[],
        context={},
        hitl_mode="sync",
    )
    assert req.hitl_mode == "sync"


def test_agent_request_defaults_hitl_mode_to_sync():
    req = AgentRequest(
        deliverable_slug="s",
        deliverable_type="spec",
        outer_state="spec",
        compliance_profile="prototype",
    )
    assert req.hitl_mode == "sync"
    assert req.upstream_artifacts == []
    assert req.context == {}


def test_agent_request_rejects_unknown_deliverable_type():
    with pytest.raises(ValidationError):
        AgentRequest(
            deliverable_slug="s",
            deliverable_type="bogus",
            outer_state="spec",
            compliance_profile="prototype",
        )


def test_agent_result_invariant_ok_matches_verdict_pass():
    r = AgentResult(ok=True, output="/x", reasons=[], verdict="GateSpecPass")
    assert r.ok == r.verdict.endswith("Pass")


def test_agent_result_invariant_rejects_mismatch():
    with pytest.raises(ValidationError):
        AgentResult(ok=True, output="/x", reasons=[], verdict="GateSpecFail")


def test_agent_result_allows_missing_optional_fields():
    r = AgentResult(ok=True, output="/x", reasons=[])
    assert r.verdict is None
    assert r.artifacts == []
    assert r.gate_category is None
    assert r.inner_state_final is None
    assert r.metrics == {}


def test_agent_envelope_wraps_result():
    jid = uuid4()
    env = AgentEnvelope(
        job_id=jid,
        result=AgentResult(ok=True, output="/x", reasons=[]),
    )
    assert env.job_id == jid
    assert env.result.ok is True


def test_artifact_ref_model_round_trip():
    ref = ArtifactRef(
        deliverable_slug="s",
        artifact_ref="spec/v1",
        storage_path="/artifacts/s/spec/v1/content.bin",
    )
    assert ref.model_dump()["deliverable_slug"] == "s"


def test_agent_envelope_job_id_is_uuid():
    env = AgentEnvelope(
        job_id=uuid4(),
        result=AgentResult(ok=False, output="", reasons=["x"]),
    )
    assert isinstance(env.job_id, UUID)
