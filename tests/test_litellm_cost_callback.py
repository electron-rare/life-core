"""LiteLLM callback must write a cost row (observable via our wrapper)."""
from __future__ import annotations

from unittest.mock import MagicMock


def test_cost_callback_records_row():
    from life_core.router.providers import litellm_cost_callback

    writer = MagicMock()
    litellm_cost_callback(
        kwargs={"model": "openai/qwen-14b-awq-kxkm"},
        completion_response={
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "_hidden_params": {"response_cost": 0.0002},
        },
        start_time=0.0,
        end_time=0.1,
        writer=writer,
    )
    writer.assert_called_once()
    row = writer.call_args.args[0]
    assert row["model"] == "openai/qwen-14b-awq-kxkm"
    assert row["cost_usd"] == 0.0002
    assert row["tokens_in"] == 10
    assert row["tokens_out"] == 5
