"""SQLAlchemy 2.x models for the inner_trace schema (HITL agent traces)."""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    __table_args__ = {"schema": "inner_trace"}


class AgentRun(Base):
    __tablename__ = "agent_run"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    deliverable_slug = Column(String, nullable=False)
    deliverable_type = Column(String, nullable=False)
    role = Column(String, nullable=False)
    outer_state_at_start = Column(String, nullable=False)
    compliance_profile = Column(String)
    inner_state = Column(String, nullable=False, server_default=text("'DRAFT'"))
    verdict = Column(String)
    gate_category = Column(String)
    started_at = Column(DateTime(timezone=True), server_default=text("now()"))
    completed_at = Column(DateTime(timezone=True))
    time_in_loop_seconds = Column(Integer, server_default=text("0"))
    human_time_seconds = Column(Integer, server_default=text("0"))
    metadata_ = Column("metadata", JSON)

    __table_args__ = (
        Index("idx_agent_run_deliverable", "deliverable_slug"),
        {"schema": "inner_trace"},
    )


class Artifact(Base):
    __tablename__ = "artifact"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    deliverable_slug = Column(String, nullable=False)
    type = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    source = Column(String)
    metadata_ = Column("metadata", JSON)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        CheckConstraint(
            "type IN ('spec','hardware','firmware','compliance','bom','simulation')",
            name="ck_artifact_type",
        ),
        CheckConstraint(
            "source IN ('llm','human','hybrid')",
            name="ck_artifact_source",
        ),
        UniqueConstraint(
            "deliverable_slug",
            "type",
            "version",
            name="uq_artifact_slug_type_version",
        ),
        {"schema": "inner_trace"},
    )


class GenerationRun(Base):
    __tablename__ = "generation_run"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("inner_trace.agent_run.id"),
        nullable=False,
    )
    attempt_number = Column(Integer, nullable=False)
    llm_model = Column(String)
    prompt_template = Column(String)
    context_snapshot = Column(JSON)
    output_artifact_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("inner_trace.artifact.id"),
    )
    status = Column(String)
    error = Column(String)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    cost_usd = Column(Numeric(10, 4))
    user_id = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('success','failed','partial')",
            name="ck_genrun_status",
        ),
        UniqueConstraint(
            "agent_run_id",
            "attempt_number",
            name="uq_genrun_agent_attempt",
        ),
        Index("idx_genrun_user_id", "user_id"),
        {"schema": "inner_trace"},
    )


class Relation(Base):
    __tablename__ = "relation"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    from_id = Column(PGUUID(as_uuid=True), nullable=False)
    from_kind = Column(String, nullable=False)
    to_id = Column(PGUUID(as_uuid=True), nullable=False)
    to_kind = Column(String, nullable=False)
    relation_type = Column(String, nullable=False)
    metadata_ = Column("metadata", JSON)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        CheckConstraint(
            "relation_type IN ('derives_from','reviews','simulates','tests',"
            "'supersedes','is_gold_standard_for')",
            name="ck_relation_type",
        ),
        UniqueConstraint(
            "from_id",
            "to_id",
            "relation_type",
            name="uq_relation",
        ),
        {"schema": "inner_trace"},
    )


class Evaluation(Base):
    __tablename__ = "evaluation"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    llm_artifact_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("inner_trace.artifact.id"),
    )
    human_artifact_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("inner_trace.artifact.id"),
    )
    comparator = Column(String)
    score = Column(Numeric(5, 4))
    details = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        CheckConstraint(
            "comparator IN ('spec_coverage','hardware_diff',"
            "'firmware_behavior','simulation_diff')",
            name="ck_eval_comparator",
        ),
        {"schema": "inner_trace"},
    )
