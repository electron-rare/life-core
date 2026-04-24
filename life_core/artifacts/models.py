"""Pydantic reference model for an immutable stored artifact."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel


class ArtifactRef(BaseModel):
    """External handle to a written artifact (DB row + file on disk)."""

    id: UUID
    deliverable_slug: str
    type: str
    version: int
    storage_path: Path
    content_hash: str
    source: str  # 'llm' | 'human' | 'hybrid'
