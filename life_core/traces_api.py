"""Traces API — proxy to Jaeger for the cockpit."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

logger = logging.getLogger("life_core.traces_api")

traces_router = APIRouter(prefix="/traces", tags=["Traces"])

JAEGER_URL = "http://jaeger:16686"


@traces_router.get("/services")
async def list_services():
    """List traced services from Jaeger."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{JAEGER_URL}/api/services")
            return resp.json()
    except Exception as e:
        return {"data": [], "error": str(e)}


@traces_router.get("/recent")
async def recent_traces(service: str = "life-core", limit: int = 20):
    """Get recent traces from Jaeger."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{JAEGER_URL}/api/traces",
                params={"service": service, "limit": limit, "lookback": "1h"},
            )
            return resp.json()
    except Exception as e:
        return {"data": [], "error": str(e)}


def _fetch_inner_traces(limit: int = 20) -> list[dict]:
    """Read recent rows from inner_trace.generation_run joined with agent_run."""
    import os

    from sqlalchemy import create_engine, text

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return []
    engine = create_engine(dsn, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT g.id::text AS id,
                       g.agent_run_id::text AS agent_run_id,
                       g.llm_model,
                       g.tokens_in,
                       g.tokens_out,
                       g.cost_usd,
                       g.status,
                       g.started_at
                FROM inner_trace.generation_run g
                ORDER BY g.started_at DESC NULLS LAST
                LIMIT :lim
                """
            ),
            {"lim": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


@traces_router.get("/inner")
async def inner_traces(limit: int = 20):
    """Return recent inner_trace generation_run rows for cockpit display."""
    try:
        rows = _fetch_inner_traces(limit=limit)
        return {"data": rows}
    except Exception as exc:
        logger.warning("inner_traces failed: %s", exc)
        return {"data": [], "error": str(exc)}
