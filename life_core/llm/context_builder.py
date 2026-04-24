"""Compose the prompt-context dict that feeds Jinja2 generator templates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from life_core.artifacts.models import ArtifactRef


def build_context(
    *,
    deliverable_slug: str,
    deliverable_type: str,
    upstream_artifacts: list[ArtifactRef],
    brief: str,
    constraints: list[str],
    human_feedback: str | None = None,
) -> dict[str, Any]:
    """Compose the dict that feeds into Jinja2 prompts.

    Reads upstream artifact bytes via ``Path(a.storage_path).read_text()``
    so that Jinja2 templates can inline the content directly.
    """
    return {
        "deliverable_slug": deliverable_slug,
        "deliverable_type": deliverable_type,
        "brief": brief,
        "constraints": constraints,
        "human_feedback": human_feedback,
        "upstream": [
            {
                "deliverable_slug": a.deliverable_slug,
                "type": a.type,
                "version": a.version,
                "content": Path(a.storage_path).read_text(errors="replace"),
            }
            for a in upstream_artifacts
        ],
    }
