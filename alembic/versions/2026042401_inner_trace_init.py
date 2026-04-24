"""inner_trace schema and tables

Revision ID: bd9c3a0a7276
Revises:
Create Date: 2026-04-24 17:30:57.023683

Creates the ``inner_trace`` PostgreSQL schema and its 5 tables that record
the HITL (human-in-the-loop) agent trace : agent_run, artifact,
generation_run, relation, evaluation. See life_core/inner_trace/models.py
for the ORM definitions.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PGUUID


# revision identifiers, used by Alembic.
revision: str = "bd9c3a0a7276"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS inner_trace")

    op.create_table(
        "agent_run",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("deliverable_slug", sa.String(), nullable=False),
        sa.Column("deliverable_type", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("outer_state_at_start", sa.String(), nullable=False),
        sa.Column("compliance_profile", sa.String(), nullable=True),
        sa.Column(
            "inner_state",
            sa.String(),
            nullable=False,
            server_default=sa.text("'DRAFT'"),
        ),
        sa.Column("verdict", sa.String(), nullable=True),
        sa.Column("gate_category", sa.String(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "time_in_loop_seconds",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=True,
        ),
        sa.Column(
            "human_time_seconds",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=True,
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        schema="inner_trace",
    )
    op.create_index(
        "idx_agent_run_deliverable",
        "agent_run",
        ["deliverable_slug"],
        schema="inner_trace",
    )

    op.create_table(
        "artifact",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("deliverable_slug", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "type IN ('spec','hardware','firmware','compliance','bom','simulation')",
            name="ck_artifact_type",
        ),
        sa.CheckConstraint(
            "source IN ('llm','human','hybrid')",
            name="ck_artifact_source",
        ),
        sa.UniqueConstraint(
            "deliverable_slug",
            "type",
            "version",
            name="uq_artifact_slug_type_version",
        ),
        schema="inner_trace",
    )

    op.create_table(
        "generation_run",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("agent_run_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("llm_model", sa.String(), nullable=True),
        sa.Column("prompt_template", sa.String(), nullable=True),
        sa.Column("context_snapshot", sa.JSON(), nullable=True),
        sa.Column("output_artifact_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(10, 4), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["inner_trace.agent_run.id"],
        ),
        sa.ForeignKeyConstraint(
            ["output_artifact_id"],
            ["inner_trace.artifact.id"],
        ),
        sa.CheckConstraint(
            "status IN ('success','failed','partial')",
            name="ck_genrun_status",
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            "attempt_number",
            name="uq_genrun_agent_attempt",
        ),
        schema="inner_trace",
    )

    op.create_table(
        "relation",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("from_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("from_kind", sa.String(), nullable=False),
        sa.Column("to_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("to_kind", sa.String(), nullable=False),
        sa.Column("relation_type", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "relation_type IN ('derives_from','reviews','simulates','tests',"
            "'supersedes','is_gold_standard_for')",
            name="ck_relation_type",
        ),
        sa.UniqueConstraint(
            "from_id",
            "to_id",
            "relation_type",
            name="uq_relation",
        ),
        schema="inner_trace",
    )

    op.create_table(
        "evaluation",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("llm_artifact_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("human_artifact_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("comparator", sa.String(), nullable=True),
        sa.Column("score", sa.Numeric(5, 4), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["llm_artifact_id"],
            ["inner_trace.artifact.id"],
        ),
        sa.ForeignKeyConstraint(
            ["human_artifact_id"],
            ["inner_trace.artifact.id"],
        ),
        sa.CheckConstraint(
            "comparator IN ('spec_coverage','hardware_diff',"
            "'firmware_behavior','simulation_diff')",
            name="ck_eval_comparator",
        ),
        schema="inner_trace",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP SCHEMA IF EXISTS inner_trace CASCADE")
