"""Tests for life_core.traceability.service — inner DAG helpers."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from life_core.inner_trace.models import AgentRun, Relation
from life_core.traceability.service import lineage, link, runs_for_deliverable


def test_link_inserts_relation_row():
    session = MagicMock()
    from_id = uuid4()
    to_id = uuid4()

    link(
        session,
        from_id=from_id,
        from_kind="agent_run",
        to_id=to_id,
        to_kind="artifact",
        relation_type="derives_from",
    )

    assert session.add.called
    (added_row,), _kwargs = session.add.call_args
    assert isinstance(added_row, Relation)
    assert added_row.from_id == from_id
    assert added_row.from_kind == "agent_run"
    assert added_row.to_id == to_id
    assert added_row.to_kind == "artifact"
    assert added_row.relation_type == "derives_from"


def test_runs_for_deliverable_queries_agent_run():
    session = MagicMock()
    sentinel_runs = [MagicMock(spec=AgentRun), MagicMock(spec=AgentRun)]
    scalars = MagicMock()
    scalars.all.return_value = sentinel_runs
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars
    session.execute.return_value = execute_result

    result = runs_for_deliverable(session, "kxkm-batt-16ch")

    assert result == sentinel_runs
    assert session.execute.called
    (stmt,), _kwargs = session.execute.call_args
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "agent_run" in compiled
    assert "kxkm-batt-16ch" in compiled


def test_lineage_returns_empty_list_without_ids():
    session = MagicMock()

    result = lineage(session, [])

    assert result == []
    assert not session.execute.called


def test_lineage_queries_relations_with_in_clause():
    session = MagicMock()
    sentinel_rels = [MagicMock(spec=Relation)]
    scalars = MagicMock()
    scalars.all.return_value = sentinel_rels
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars
    session.execute.return_value = execute_result

    node_id = uuid4()
    result = lineage(session, [node_id])

    assert result == sentinel_rels
    (stmt,), _ = session.execute.call_args
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "relation" in compiled
    assert node_id.hex in compiled
