"""State-machine tests (T1.8b)."""
from __future__ import annotations

import pytest

from life_core.agents.state_machine import InnerState, transition


def test_draft_to_review_on_generate_ok():
    assert transition(InnerState.DRAFT, "generate_ok") == InnerState.REVIEW


def test_draft_stays_on_reprompt():
    assert transition(InnerState.DRAFT, "reprompt") == InnerState.DRAFT


def test_draft_to_timeout():
    assert transition(InnerState.DRAFT, "timeout") == InnerState.TIMEOUT


def test_review_to_approved_on_approve():
    assert transition(InnerState.REVIEW, "approve") == InnerState.APPROVED


def test_review_to_rejected_on_reject():
    assert transition(InnerState.REVIEW, "reject") == InnerState.REJECTED


def test_review_to_approved_on_edit():
    assert transition(InnerState.REVIEW, "edit") == InnerState.APPROVED


def test_review_to_draft_on_reprompt():
    assert transition(InnerState.REVIEW, "reprompt") == InnerState.DRAFT


def test_review_to_timeout():
    assert transition(InnerState.REVIEW, "timeout") == InnerState.TIMEOUT


def test_invalid_transition_raises():
    with pytest.raises(ValueError):
        transition(InnerState.APPROVED, "approve")


def test_invalid_event_raises():
    with pytest.raises(ValueError):
        transition(InnerState.DRAFT, "bogus")


def test_terminal_state_has_no_outgoing_edges():
    for terminal in (InnerState.APPROVED, InnerState.REJECTED, InnerState.TIMEOUT):
        for event in (
            "approve", "reject", "edit", "reprompt", "generate_ok", "timeout"
        ):
            with pytest.raises(ValueError):
                transition(terminal, event)
