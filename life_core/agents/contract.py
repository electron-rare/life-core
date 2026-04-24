"""Pydantic contracts for the inner HITL agents module.

``AgentEnvelope`` is the two-level shape engine's ``LifeCoreClient.runAgent``
consumes: ``{ job_id, result: AgentResult }``. ``AgentResult`` keeps the
original engine-consumed core fields (ok / output / reasons) and carries a
set of enriched optional fields used by life-web only.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ArtifactRef(BaseModel):
    """Upstream artifact reference passed into an agent request."""

    deliverable_slug: str
    artifact_ref: str
    storage_path: str


class AgentRequest(BaseModel):
    """Payload accepted by ``POST /agents/{role}/run``."""

    deliverable_slug: str
    deliverable_type: Literal[
        "spec", "hardware", "firmware", "compliance", "bom", "simulation"
    ]
    outer_state: Literal["intake", "spec", "impl", "ship"]
    compliance_profile: str
    upstream_artifacts: list[ArtifactRef] = Field(default_factory=list)
    context: dict = Field(default_factory=dict)
    hitl_mode: Literal["sync", "async"] = "sync"


class AgentResult(BaseModel):
    """Inner agent run result.

    Core engine-consumed fields (``ok`` / ``output`` / ``reasons``) must stay
    stable. Optional fields enrich the payload for the life-web cockpit.
    """

    ok: bool
    output: str
    reasons: list[str] = Field(default_factory=list)
    agent_run_id: UUID | None = None
    verdict: Literal[
        "GateSpecPass", "GateSpecFail", "GateImplPass", "GateImplFail"
    ] | None = None
    gate_category: Literal["compliance", "test", "build"] | None = None
    artifacts: list[dict] = Field(default_factory=list)
    inner_state_final: Literal["APPROVED", "REJECTED", "TIMEOUT"] | None = None
    metrics: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ok_matches_verdict(self) -> "AgentResult":
        if self.verdict is not None and self.ok != self.verdict.endswith("Pass"):
            raise ValueError("ok must equal verdict.endswith('Pass')")
        return self


class AgentEnvelope(BaseModel):
    """Two-level envelope consumed by engine ``LifeCoreClient.runAgent()``."""

    job_id: UUID
    result: AgentResult
