"""V1.8 add inner_trace.cost_ledger table."""
from alembic import op
import sqlalchemy as sa

revision = "v18_add_cost_ledger"
down_revision = "v18_add_user_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cost_ledger",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()")),
        schema="inner_trace",
    )
    op.create_index(
        "idx_cost_ledger_created_model",
        "cost_ledger",
        ["created_at", "model"],
        schema="inner_trace",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_cost_ledger_created_model",
        table_name="cost_ledger", schema="inner_trace",
    )
    op.drop_table("cost_ledger", schema="inner_trace")
