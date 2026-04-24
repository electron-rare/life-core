"""Compute the next monotonic version for (deliverable_slug, type)."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from life_core.inner_trace.models import Artifact


def next_version(
    session: Session, deliverable_slug: str, artifact_type: str
) -> int:
    """Return 1 + max(version) for (slug, type), or 1 if none exists yet."""
    stmt = select(func.max(Artifact.version)).where(
        Artifact.deliverable_slug == deliverable_slug,
        Artifact.type == artifact_type,
    )
    current = session.execute(stmt).scalar()
    return 1 if current is None else current + 1
