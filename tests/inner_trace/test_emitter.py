"""Tests for the inner_trace emitter helper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from life_core.inner_trace.emitter import TraceEmitter


def test_emitter_records_agent_run():
    session = MagicMock()
    emitter = TraceEmitter(session_factory=lambda: session)
    run_id = emitter.record_agent_run(
        deliverable_slug="spec-abc",
        deliverable_type="spec",
        role="llm",
        outer_state_at_start="OPEN",
    )
    assert run_id is not None
    assert session.add.called
    assert session.commit.called


def test_emitter_records_generation_run():
    session = MagicMock()
    emitter = TraceEmitter(session_factory=lambda: session)
    gen_id = emitter.record_generation_run(
        agent_run_id="00000000-0000-0000-0000-000000000001",
        attempt_number=1,
        llm_model="openai/qwen-14b-awq-kxkm",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.0012,
        status="success",
    )
    assert gen_id is not None
    assert session.add.called
    assert session.commit.called


def test_emitter_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("INNER_TRACE_ENABLED", "false")
    emitter = TraceEmitter(session_factory=lambda: MagicMock())
    assert emitter.record_agent_run(
        deliverable_slug="x", deliverable_type="spec",
        role="llm", outer_state_at_start="OPEN",
    ) is None
