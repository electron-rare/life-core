"""E2E smoke: after a chat call, a row exists in inner_trace.generation_run.

Marked integration — skipped when DATABASE_URL is not set.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_round_trip_writes_inner_trace_row():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set, skipping integration smoke")

    from sqlalchemy import create_engine, text
    from life_core.inner_trace.emitter import TraceEmitter
    from life_core.services.chat import ChatService

    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        before = conn.execute(
            text("SELECT COUNT(*) FROM inner_trace.generation_run"),
        ).scalar_one()

    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    emitter = TraceEmitter(session_factory=Session)

    router = MagicMock()
    router.send = AsyncMock(return_value=MagicMock(
        content="ok",
        usage={"prompt_tokens": 3, "completion_tokens": 1},
        model="openai/qwen-14b-awq-kxkm",
        cost_usd=0.00001,
    ))
    svc = ChatService(router=router, cache=MagicMock(), trace_emitter=emitter)

    await svc.chat(
        messages=[{"role": "user", "content": "ping"}],
        model="openai/qwen-14b-awq-kxkm",
        deliverable_slug="e2e-smoke",
    )

    with engine.connect() as conn:
        after = conn.execute(
            text("SELECT COUNT(*) FROM inner_trace.generation_run"),
        ).scalar_one()
    assert after == before + 1
