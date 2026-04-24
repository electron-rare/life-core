"""Inner DAG traceability service.

Thin SQLAlchemy 2.x wrappers over the ``inner_trace`` schema. Exposes:

- ``link(...)``            — insert a ``Relation`` row between two nodes.
- ``runs_for_deliverable`` — list ``AgentRun`` rows scoped to a deliverable.
- ``lineage(...)``         — list ``Relation`` rows touching a set of nodes.

The shape is deliberately 1-for-1 mappable onto Neo4j (ADR-004).
"""
from __future__ import annotations

from typing import Iterable, Sequence
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from life_core.inner_trace.models import AgentRun, Relation


def link(
    session: Session,
    *,
    from_id: UUID,
    from_kind: str,
    to_id: UUID,
    to_kind: str,
    relation_type: str,
    metadata: dict | None = None,
) -> Relation:
    """Insert a ``Relation`` row connecting two inner_trace nodes."""
    row = Relation(
        from_id=from_id,
        from_kind=from_kind,
        to_id=to_id,
        to_kind=to_kind,
        relation_type=relation_type,
        metadata_=metadata,
    )
    session.add(row)
    return row


def runs_for_deliverable(
    session: Session, deliverable_slug: str
) -> Sequence[AgentRun]:
    """Return all ``AgentRun`` rows scoped to a deliverable_slug."""
    stmt = select(AgentRun).where(AgentRun.deliverable_slug == deliverable_slug)
    return session.execute(stmt).scalars().all()


def lineage(
    session: Session, node_ids: Iterable[UUID]
) -> Sequence[Relation]:
    """Return relations where from_id or to_id is in ``node_ids``."""
    ids = list(node_ids)
    if not ids:
        return []
    stmt = select(Relation).where(
        or_(Relation.from_id.in_(ids), Relation.to_id.in_(ids))
    )
    return session.execute(stmt).scalars().all()
