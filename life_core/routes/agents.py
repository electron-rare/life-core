"""FastAPI router exposing LLM agents under /agents/{role}/run."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from life_core.agents.qa import QaAgent
from life_core.agents.spec import SpecAgent

router = APIRouter(prefix="/agents", tags=["agents"])

_AGENTS: dict[str, Any] = {
    "spec": SpecAgent(),
    "qa": QaAgent(),
}


@router.post("/{role}/run")
async def run_agent(role: str, payload: dict[str, Any]) -> dict[str, Any]:
    agent = _AGENTS.get(role)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"unknown agent role: {role}")
    job_id = f"job-{uuid.uuid4()}"
    result = await agent.run(payload)
    return {"job_id": job_id, "result": result.__dict__}
