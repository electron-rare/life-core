"""Smoke tests for the inner_trace SQLAlchemy models."""
from life_core.inner_trace.models import (
    AgentRun,
    Artifact,
    Base,
    Evaluation,
    GenerationRun,
    Relation,
)


def test_tables_in_schema_inner_trace() -> None:
    names = {t.name for t in Base.metadata.tables.values()}
    assert names == {
        "agent_run",
        "artifact",
        "generation_run",
        "relation",
        "evaluation",
    }
    for table in Base.metadata.tables.values():
        assert table.schema == "inner_trace"


def test_models_importable() -> None:
    # Ensures no import-time errors in the 5 models.
    assert AgentRun.__tablename__ == "agent_run"
    assert Artifact.__tablename__ == "artifact"
    assert GenerationRun.__tablename__ == "generation_run"
    assert Relation.__tablename__ == "relation"
    assert Evaluation.__tablename__ == "evaluation"
