"""V1.8 add user_id column to inner_trace.generation_run."""
from alembic import op
import sqlalchemy as sa

revision = "v18_add_user_id"
down_revision = "bd9c3a0a7276"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generation_run",
        sa.Column("user_id", sa.String(), nullable=True),
        schema="inner_trace",
    )
    op.create_index(
        "idx_genrun_user_id",
        "generation_run",
        ["user_id"],
        schema="inner_trace",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_genrun_user_id", table_name="generation_run", schema="inner_trace",
    )
    op.drop_column("generation_run", "user_id", schema="inner_trace")
