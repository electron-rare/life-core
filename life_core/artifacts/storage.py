"""Immutable versioned artifact storage on a local (volume-mounted) path.

Layout:
    <volume_root>/<deliverable_slug>/<type>/v<N>/content.<hash_prefix>.bin
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from life_core.inner_trace.models import Artifact

from .hash import content_hash
from .models import ArtifactRef
from .versioning import next_version


def write(
    session: Session,
    volume_root: Path,
    deliverable_slug: str,
    artifact_type: str,
    data: bytes,
    *,
    source: str,
    metadata: dict | None = None,
) -> ArtifactRef:
    """Write bytes to an immutable versioned path and insert the DB row.

    Returns a Pydantic ArtifactRef describing the written artifact. The
    caller is responsible for committing the surrounding transaction; we
    only flush so the row's identity becomes available.
    """
    version = next_version(session, deliverable_slug, artifact_type)
    h = content_hash(data)
    folder = Path(volume_root) / deliverable_slug / artifact_type / f"v{version}"
    folder.mkdir(parents=True, exist_ok=False)
    file_path = folder / f"content.{h[:8]}.bin"
    file_path.write_bytes(data)

    row = Artifact(
        id=uuid4(),
        deliverable_slug=deliverable_slug,
        type=artifact_type,
        version=version,
        storage_path=str(file_path),
        content_hash=h,
        source=source,
        metadata_=metadata,
    )
    session.add(row)
    session.flush()  # immediate INSERT; commit is caller's responsibility
    return ArtifactRef(
        id=row.id,
        deliverable_slug=deliverable_slug,
        type=artifact_type,
        version=version,
        storage_path=file_path,
        content_hash=h,
        source=source,
    )


def read(ref: ArtifactRef) -> bytes:
    """Read back bytes referenced by an ArtifactRef."""
    return Path(ref.storage_path).read_bytes()
