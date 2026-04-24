"""Langfuse bridge must forward inner_trace generation_run payloads."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_forward_generation_run_to_langfuse(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.saillant.cc")

    from life_core import langfuse_tracing

    fake_client = MagicMock()
    fake_trace = MagicMock()
    fake_client.trace.return_value = fake_trace

    with patch.object(langfuse_tracing, "_langfuse", fake_client):
        langfuse_tracing.forward_generation_run(
            generation_run_id="gen-1",
            agent_run_id="run-1",
            deliverable_slug="spec-x",
            llm_model="openai/qwen-14b-awq-kxkm",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
            user_id="user-42",
        )

    fake_client.trace.assert_called_once()
    fake_trace.generation.assert_called_once()
    kwargs = fake_trace.generation.call_args.kwargs
    assert kwargs["model"] == "openai/qwen-14b-awq-kxkm"
    assert kwargs["usage"]["input"] == 10
    assert kwargs["usage"]["output"] == 5
