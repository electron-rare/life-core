"""FastAPI router exposing the inner DAG via HTTP.

``GET /traceability/graph?deliverable_slug=<slug>`` returns the runs + relations
scoped to the given deliverable, shaped for the life-web DAG viewer
(Sprint 1 P5). The payload is a thin JSON projection of the SQLAlchemy rows
and is mappable onto Neo4j node/edge shapes (ADR-004).
"""
from __future__ import annotations

from typing import Iterator

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from life_core.traceability.service import lineage, runs_for_deliverable

router = APIRouter(prefix="/traceability", tags=["traceability"])


def get_session() -> Iterator[Session]:
    """Placeholder DB session dependency.

    Resolved via ``app.dependency_overrides[get_session]`` in tests and by the
    real SessionLocal factory in production wiring.
    """
    raise NotImplementedError(
        "traceability.get_session must be overridden via "
        "app.dependency_overrides or a production SessionLocal wiring."
    )


def _serialize_run(run) -> dict:
    return {
        "id": str(run.id) if run.id is not None else None,
        "deliverable_slug": run.deliverable_slug,
        "deliverable_type": run.deliverable_type,
        "role": run.role,
        "outer_state_at_start": run.outer_state_at_start,
        "inner_state": run.inner_state,
        "verdict": run.verdict,
        "gate_category": run.gate_category,
    }


def _serialize_relation(rel) -> dict:
    return {
        "id": str(rel.id) if rel.id is not None else None,
        "from_id": str(rel.from_id),
        "from_kind": rel.from_kind,
        "to_id": str(rel.to_id),
        "to_kind": rel.to_kind,
        "relation_type": rel.relation_type,
    }


@router.get("/graph")
def get_graph(
    deliverable_slug: str = Query(..., min_length=1),
    session: Session = Depends(get_session),
) -> dict:
    """Return runs + relations for a deliverable."""
    runs = runs_for_deliverable(session, deliverable_slug)
    run_ids = [run.id for run in runs if run.id is not None]
    relations = lineage(session, run_ids)
    return {
        "runs": [_serialize_run(run) for run in runs],
        "relations": [_serialize_relation(rel) for rel in relations],
    }
