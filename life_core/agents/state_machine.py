"""Inner HITL state machine (P4 T1.8b).

Five states and eight legal transitions. ``transition()`` raises
``ValueError`` on any illegal ``(state, event)`` pair so the
orchestrator never silently drops an update.
"""
from __future__ import annotations

from enum import Enum


class InnerState(str, Enum):
    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


_TRANSITIONS: dict[tuple[InnerState, str], InnerState] = {
    (InnerState.DRAFT, "generate_ok"): InnerState.REVIEW,
    (InnerState.DRAFT, "reprompt"): InnerState.DRAFT,
    (InnerState.DRAFT, "timeout"): InnerState.TIMEOUT,
    (InnerState.REVIEW, "approve"): InnerState.APPROVED,
    (InnerState.REVIEW, "reject"): InnerState.REJECTED,
    (InnerState.REVIEW, "edit"): InnerState.APPROVED,
    (InnerState.REVIEW, "reprompt"): InnerState.DRAFT,
    (InnerState.REVIEW, "timeout"): InnerState.TIMEOUT,
}


def transition(current: InnerState, event: str) -> InnerState:
    """Return the next ``InnerState`` for ``(current, event)``.

    Raises ``ValueError`` if the pair is not in the legal table.
    """
    key = (current, event)
    if key not in _TRANSITIONS:
        raise ValueError(f"Invalid inner transition: {current} -- {event} -->")
    return _TRANSITIONS[key]
