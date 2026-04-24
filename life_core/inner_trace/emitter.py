"""Thin wrapper around the inner_trace SQLAlchemy models used by life-core services."""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Callable, Optional
from uuid import UUID, uuid4

from life_core.inner_trace.models import AgentRun, GenerationRun
from life_core.langfuse_tracing import forward_generation_run

logger = logging.getLogger("life_core.inner_trace.emitter")


def _enabled() -> bool:
    return os.environ.get("INNER_TRACE_ENABLED", "true").lower() not in (
        "false", "0", "no",
    )


class TraceEmitter:
    """Emit rows into inner_trace.agent_run and inner_trace.generation_run."""

    def __init__(self, session_factory: Callable):
        self._session_factory = session_factory

    def record_agent_run(
        self,
        deliverable_slug: str,
        deliverable_type: str,
        role: str,
        outer_state_at_start: str,
        compliance_profile: Optional[str] = None,
    ) -> Optional[UUID]:
        if not _enabled():
            return None
        run_id = uuid4()
        try:
            session = self._session_factory()
            row = AgentRun(
                id=run_id,
                deliverable_slug=deliverable_slug,
                deliverable_type=deliverable_type,
                role=role,
                outer_state_at_start=outer_state_at_start,
                compliance_profile=compliance_profile,
            )
            session.add(row)
            session.commit()
            return run_id
        except Exception as exc:
            logger.warning("record_agent_run failed: %s", exc)
            return None

    def record_generation_run(
        self,
        agent_run_id: str,
        attempt_number: int,
        llm_model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        status: str,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[UUID]:
        if not _enabled():
            return None
        gen_id = uuid4()
        try:
            session = self._session_factory()
            row = GenerationRun(
                id=gen_id,
                agent_run_id=agent_run_id,
                attempt_number=attempt_number,
                llm_model=llm_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=Decimal(str(cost_usd)),
                status=status,
                error=error,
                user_id=user_id,
            )
            session.add(row)
            session.commit()
            try:
                forward_generation_run(
                    generation_run_id=str(gen_id),
                    agent_run_id=str(agent_run_id),
                    deliverable_slug="",
                    llm_model=llm_model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_usd,
                    user_id=user_id or os.environ.get("INNER_TRACE_USER_ID"),
                )
            except Exception as exc:
                logger.warning("langfuse forward failed: %s", exc)
            return gen_id
        except Exception as exc:
            logger.warning("record_generation_run failed: %s", exc)
            return None
