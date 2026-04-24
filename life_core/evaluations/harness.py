"""Evaluation harness (P4 T1.9c).

Given two :class:`AgentRun` ids (LLM and human gold standard), pick the
latest :class:`Artifact` of each side, run the comparator matching the
artifact's type in a thread executor (so sync comparators do not block
the event loop), and persist an :class:`Evaluation` row.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from life_core.inner_trace.models import AgentRun, Artifact, Evaluation

from .comparators import (
    firmware_behavior_compare,
    hardware_diff_compare,
    simulation_diff_compare,
    spec_coverage_compare,
)

_COMPARATORS = {
    "spec": (
        "spec_coverage",
        lambda h, l: spec_coverage_compare(
            Path(h).read_text(), Path(l).read_text()
        ),
    ),
    "hardware": (
        "hardware_diff",
        lambda h, l: hardware_diff_compare(h, l),
    ),
    "firmware": (
        "firmware_behavior",
        lambda h, l: firmware_behavior_compare(h, l),
    ),
    "simulation": (
        "simulation_diff",
        lambda h, l: simulation_diff_compare(h, l),
    ),
    "compliance": (
        "spec_coverage",
        lambda h, l: spec_coverage_compare(
            Path(h).read_text(), Path(l).read_text()
        ),
    ),
}


async def run_evaluation(
    session: Session,
    llm_agent_run_id: UUID,
    human_agent_run_id: UUID,
) -> list[Evaluation]:
    """Evaluate the artifact pair for two agent runs."""

    llm_run = session.get(AgentRun, llm_agent_run_id)
    human_run = session.get(AgentRun, human_agent_run_id)
    if not llm_run or not human_run:
        raise ValueError("agent_run(s) not found")

    llm_art = (
        session.execute(
            select(Artifact)
            .where(
                Artifact.deliverable_slug == llm_run.deliverable_slug,
                Artifact.source == "llm",
            )
            .order_by(Artifact.version.desc())
        )
        .scalars()
        .first()
    )
    human_art = (
        session.execute(
            select(Artifact)
            .where(
                Artifact.deliverable_slug == human_run.deliverable_slug,
                Artifact.source == "human",
            )
            .order_by(Artifact.version.desc())
        )
        .scalars()
        .first()
    )
    if not (llm_art and human_art):
        raise ValueError("artifact pair missing")

    dtype = llm_art.type
    if dtype not in _COMPARATORS:
        raise ValueError(f"Unknown type {dtype}")
    name, fn = _COMPARATORS[dtype]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, fn, human_art.storage_path, llm_art.storage_path
    )

    ev = Evaluation(
        llm_artifact_id=llm_art.id,
        human_artifact_id=human_art.id,
        comparator=name,
        score=result["score"],
        details=result.get("details"),
    )
    session.add(ev)
    session.flush()
    return [ev]
