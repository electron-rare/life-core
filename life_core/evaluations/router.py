"""FastAPI router for the evaluations module (P4 T1.12a).

Exposes a single endpoint::

    POST /evaluations/run?llm_agent_run_id=<uuid>&human_agent_run_id=<uuid>

``get_session`` is imported defensively, mirroring
:mod:`life_core.agents.router` — tests override it via
``app.dependency_overrides``.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

try:  # pragma: no cover - wired in a later sprint
    from life_core.db import get_session  # type: ignore
except ImportError:  # pragma: no cover

    def get_session():  # type: ignore[misc]
        raise NotImplementedError("wire a real SessionLocal")


from .harness import run_evaluation

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run")
async def run(
    llm_agent_run_id: UUID,
    human_agent_run_id: UUID,
    session: Session = Depends(get_session),
) -> dict:
    try:
        evs = await run_evaluation(
            session, llm_agent_run_id, human_agent_run_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "evaluations": [
            {
                "id": str(e.id),
                "comparator": e.comparator,
                "score": float(e.score),
                "details": e.details,
            }
            for e in evs
        ],
    }
