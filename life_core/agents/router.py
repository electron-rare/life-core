"""FastAPI router for the inner HITL agents module (P4 T1.8d).

Exposes three endpoints::

    POST /agents/{role}/run                 → AgentEnvelope
    POST /agents/{role}/decide/{agent_run_id}?decision=approve|reject|edit|reprompt
    GET  /agents/runs/{agent_run_id}

``get_session`` is imported defensively: ``life_core.db`` does not yet
exist on ``feat/inner-trace-schema`` — the fallback raises
``NotImplementedError`` if the app is wired without a real session
factory. Tests either bypass DB calls via mocks or use FastAPI
``dependency_overrides``.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

try:  # pragma: no cover - wired in a later sprint
    from life_core.db import get_session  # type: ignore
except ImportError:  # pragma: no cover

    def get_session():  # type: ignore[misc]
        raise NotImplementedError("wire a real SessionLocal")


from .contract import AgentEnvelope, AgentRequest
from .orchestrator import run_agent

router = APIRouter(prefix="/agents", tags=["agents"])
_VOLUME_ROOT = Path("/artifacts")
_ALLOWED_ROLES = {"spec", "impl", "qa"}
_ALLOWED_DECISIONS = {"approve", "reject", "edit", "reprompt"}


@router.post("/{role}/run", response_model=AgentEnvelope)
async def run(
    role: str,
    req: AgentRequest,
    session: Session = Depends(get_session),
) -> AgentEnvelope:
    if role not in _ALLOWED_ROLES:
        raise HTTPException(status_code=422, detail=f"Unknown role: {role}")
    return await run_agent(
        role, req, session=session, volume_root=_VOLUME_ROOT
    )


@router.post("/{role}/decide/{agent_run_id}")
def decide(
    role: str,
    agent_run_id: str,
    decision: str,
    session: Session = Depends(get_session),
) -> dict:
    from life_core.inner_trace.models import AgentRun

    if role not in _ALLOWED_ROLES:
        raise HTTPException(status_code=422, detail=f"Unknown role: {role}")
    if decision not in _ALLOWED_DECISIONS:
        raise HTTPException(status_code=422, detail="unknown decision")
    row = session.get(AgentRun, UUID(agent_run_id))
    if not row:
        raise HTTPException(status_code=404, detail="agent_run not found")
    row.inner_state = {
        "approve": "APPROVED",
        "reject": "REJECTED",
        "edit": "APPROVED",
        "reprompt": "DRAFT",
    }[decision]
    session.flush()
    return {"ok": True, "inner_state": row.inner_state}


@router.get("/runs/{agent_run_id}")
def get_run(
    agent_run_id: str,
    session: Session = Depends(get_session),
) -> dict:
    from life_core.inner_trace.models import AgentRun

    row = session.get(AgentRun, UUID(agent_run_id))
    if not row:
        raise HTTPException(status_code=404, detail="agent_run not found")
    return {
        "id": str(row.id),
        "deliverable_slug": row.deliverable_slug,
        "role": row.role,
        "inner_state": row.inner_state,
        "verdict": row.verdict,
        "started_at": row.started_at.isoformat() if row.started_at else None,
    }
