"""V1.7 Track II Task 7 — /traces endpoint (Langfuse direct)."""
from __future__ import annotations

import asyncio

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-abc")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-xyz")


@respx.mock
def test_traces_endpoint_returns_traces():
    from life_core.api import app
    from life_core.integrations import langfuse as lf

    lf._seen_trace_ids.clear()
    respx.get("https://langfuse.test/api/public/traces").mock(
        return_value=Response(
            200,
            json={
                "data": [
                    {
                        "id": "t-1",
                        "totalCost": 0.01,
                        "latency": 120,
                    },
                    {
                        "id": "t-2",
                        "totalCost": 0.02,
                        "latency": 340,
                    },
                    {
                        "id": "t-3",
                        "totalCost": 0.03,
                        "latency": 560,
                    },
                ],
                "meta": {"nextCursor": "c-2"},
            },
        )
    )

    client = TestClient(app)
    resp = client.get(
        "/traces", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["traces"]) == 3
    assert body["cursor"] == "c-2"
    assert body["traces"][0]["id"] == "t-1"


@respx.mock
def test_traces_dedup_emits_three_then_zero():
    """Three new trace ids emit three langfuse.trace events; second
    call with identical ids emits zero."""
    from life_core.events.broker import get_broker
    from life_core.events.schema import EventType
    from life_core.integrations import langfuse as lf

    async def run():
        lf._seen_trace_ids.clear()
        respx.get("https://langfuse.test/api/public/traces").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {"id": "t-1", "totalCost": 0.01, "latency": 120},
                        {"id": "t-2", "totalCost": 0.02, "latency": 340},
                        {"id": "t-3", "totalCost": 0.03, "latency": 560},
                    ],
                    "meta": {"nextCursor": None},
                },
            )
        )
        broker = get_broker()
        q = broker.subscribe()
        try:
            await lf.fetch_traces(cursor=None)
            first_round = 0
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                    if ev.type is EventType.LANGFUSE_TRACE:
                        first_round += 1
            except asyncio.TimeoutError:
                pass
            assert first_round == 3

            await lf.fetch_traces(cursor=None)
            second_round = 0
            try:
                while True:
                    ev = await asyncio.wait_for(q.get(), timeout=0.2)
                    if ev.type is EventType.LANGFUSE_TRACE:
                        second_round += 1
            except asyncio.TimeoutError:
                pass
            assert second_round == 0
        finally:
            broker.unsubscribe(q)

    asyncio.run(run())
