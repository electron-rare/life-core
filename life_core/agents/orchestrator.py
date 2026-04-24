"""Async inner HITL orchestrator (P4 T1.8c).

Wraps the generator call, writes an immutable artifact, then either
returns immediately (``hitl_mode="async"``) or polls the ``agent_run``
row until a human decision lands (``hitl_mode="sync"``). The return
value is always an ``AgentEnvelope`` so the engine's ``LifeCoreClient``
sees the expected ``{ job_id, result }`` shape.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from time import time
from uuid import uuid4

from sqlalchemy.orm import Session

from life_core.artifacts import write
from life_core.generators.base import (
    BaseGenerator,
    GenerationContext,
    GenerationOutcome,
)
from life_core.generators.firmware_generator import FirmwareGenerator
from life_core.generators.kicad_generator import KicadGenerator
from life_core.generators.spec_generator import SpecGenerator
from life_core.generators.spice_generator import SpiceGenerator
from life_core.inner_trace.models import AgentRun

from .contract import AgentEnvelope, AgentRequest, AgentResult
from .state_machine import InnerState

_REVIEW_TIMEOUT_S = 600
_POLL_INTERVAL_S = 2.0


def _pick_generator(role: str, deliverable_type: str) -> BaseGenerator:
    """Route ``(role, deliverable_type)`` to a concrete generator instance."""
    if role == "spec":
        return SpecGenerator()
    return {
        "hardware": KicadGenerator(),
        "firmware": FirmwareGenerator(),
        "simulation": SpiceGenerator(),
    }.get(deliverable_type, SpecGenerator())


async def _wait_human_decision(agent_run_id, session: Session) -> str:
    """Poll ``agent_run.inner_state`` until a terminal state or timeout."""
    deadline = time() + _REVIEW_TIMEOUT_S
    while time() < deadline:
        await asyncio.sleep(_POLL_INTERVAL_S)
        row = session.get(AgentRun, agent_run_id)
        if row and row.inner_state != InnerState.REVIEW.value:
            if row.inner_state == InnerState.APPROVED.value:
                return "approve"
            if row.inner_state == InnerState.REJECTED.value:
                return "reject"
    return "timeout"


async def run_agent(
    role: str,
    req: AgentRequest,
    *,
    session: Session,
    volume_root: Path | None,
) -> AgentEnvelope:
    """Run one inner HITL turn and return the engine-facing envelope."""
    run_id = uuid4()
    run_row = AgentRun(
        id=run_id,
        deliverable_slug=req.deliverable_slug,
        deliverable_type=req.deliverable_type,
        role=role,
        outer_state_at_start=req.outer_state,
        compliance_profile=req.compliance_profile,
        inner_state=InnerState.DRAFT.value,
    )
    session.add(run_row)
    session.flush()

    gen = _pick_generator(role, req.deliverable_type)
    ctx = GenerationContext(
        deliverable_slug=req.deliverable_slug,
        deliverable_type=req.deliverable_type,
        prompt_template=(
            f"{req.deliverable_type if role == 'impl' else 'spec'}.j2"
        ),
        llm_model=(
            f"mascarade-{req.deliverable_type}"
            if role == "impl"
            else "mascarade-spec"
        ),
        prompt_vars={
            "brief": req.context.get("brief", ""),
            "constraints": req.context.get("constraints", []),
            "upstream": [a.model_dump() for a in req.upstream_artifacts],
        },
    )
    outcome: GenerationOutcome = gen.generate(ctx)

    if not outcome.ok:
        run_row.inner_state = InnerState.REJECTED.value
        run_row.verdict = "GateImplFail" if role == "impl" else "GateSpecFail"
        session.flush()
        return AgentEnvelope(
            job_id=run_id,
            result=AgentResult(
                ok=False,
                output="",
                reasons=outcome.errors,
                agent_run_id=run_id,
                verdict=run_row.verdict,
                inner_state_final=InnerState.REJECTED.value,
                metrics={"attempts": outcome.attempts},
            ),
        )

    ref = write(
        session,
        volume_root or Path("/tmp/artifacts"),
        req.deliverable_slug,
        req.deliverable_type,
        outcome.data,
        source="llm",
    )

    run_row.inner_state = InnerState.REVIEW.value
    session.flush()

    if req.hitl_mode == "async":
        return AgentEnvelope(
            job_id=run_id,
            result=AgentResult(
                ok=True,
                output=str(ref.storage_path),
                reasons=[],
                agent_run_id=run_id,
                artifacts=[
                    {
                        "type": req.deliverable_type,
                        "storage_path": str(ref.storage_path),
                        "version": ref.version,
                    }
                ],
                metrics={"attempts": outcome.attempts},
            ),
        )

    decision = await _wait_human_decision(run_id, session)
    final_state = {
        "approve": InnerState.APPROVED,
        "reject": InnerState.REJECTED,
        "timeout": InnerState.TIMEOUT,
    }[decision]
    if final_state == InnerState.APPROVED:
        verdict = "GateImplPass" if role == "impl" else "GateSpecPass"
    else:
        verdict = "GateImplFail" if role == "impl" else "GateSpecFail"
    run_row.inner_state = final_state.value
    run_row.verdict = verdict
    session.flush()
    return AgentEnvelope(
        job_id=run_id,
        result=AgentResult(
            ok=final_state == InnerState.APPROVED,
            output=str(ref.storage_path),
            reasons=[],
            agent_run_id=run_id,
            verdict=verdict,
            artifacts=[
                {
                    "type": req.deliverable_type,
                    "storage_path": str(ref.storage_path),
                    "version": ref.version,
                }
            ],
            inner_state_final=final_state.value,
            metrics={"attempts": outcome.attempts},
        ),
    )
