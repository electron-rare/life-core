"""One chat call must produce one row in inner_trace.cost_ledger (via LiteLLM callback)."""
from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_cost_ledger_grows_after_chat():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")

    from sqlalchemy import create_engine, text
    from life_core.router.providers import litellm_cost_callback

    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        before = conn.execute(
            text("SELECT COUNT(*) FROM inner_trace.cost_ledger"),
        ).scalar_one()

    litellm_cost_callback(
        kwargs={"model": "openai/qwen-14b-awq-kxkm"},
        completion_response={
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "_hidden_params": {"response_cost": 0.00001},
        },
        start_time=0.0,
        end_time=0.05,
    )

    with engine.connect() as conn:
        after = conn.execute(
            text("SELECT COUNT(*) FROM inner_trace.cost_ledger"),
        ).scalar_one()
    assert after == before + 1
