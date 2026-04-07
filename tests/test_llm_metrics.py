"""Tests for LLM call OTEL instruments."""

from __future__ import annotations


def test_create_llm_instruments():
    from life_core.telemetry import create_llm_instruments

    instruments = create_llm_instruments()
    assert "llm_calls" in instruments
    assert "llm_errors" in instruments
    assert "llm_duration" in instruments
    assert "llm_cost" in instruments
