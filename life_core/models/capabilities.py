"""V1.7 Track II — capability typing for /v1/models.

Source of truth order:
  1. explicit entry in ``CAPABILITY_OVERRIDES`` (filled at startup
     from ``KIKI_FULL_MODELS`` config when a structured mapping is
     provided).
  2. heuristic substring match on the model id:
     - contains ``embed``            -> ``["embedding"]``
     - contains ``vision``, ``-vl-`` -> ``["vision"]``
     - ends with ``-vl`` or has
       ``vl`` as a hyphen segment   -> ``["vision"]``
     - otherwise                    -> ``["chat"]``
"""
from __future__ import annotations

CAPABILITY_OVERRIDES: dict[str, list[str]] = {}


def guess_capabilities(model_id: str) -> list[str]:
    """Return the capability list for ``model_id``.

    Explicit overrides win; otherwise fall back to the heuristic.
    """
    if model_id in CAPABILITY_OVERRIDES:
        return list(CAPABILITY_OVERRIDES[model_id])
    lower = model_id.lower()
    if "embed" in lower:
        return ["embedding"]
    if "-vl-" in lower or "vision" in lower or lower.endswith("-vl"):
        return ["vision"]
    if "vl" in lower.split("-"):
        return ["vision"]
    return ["chat"]
