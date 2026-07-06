"""Add knowledge_bases and kb_user_permissions tables for Multi-KB."""
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    op.create_table("knowledge_bases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("collection_name", sa.String(200), nullable=False, unique=True),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_bases_user_id", "knowledge_bases", ["user_id"])

    op.create_table("kb_user_permissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kb_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(20), nullable=False, server_default="read"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kb_user_permissions_kb_id", "kb_user_permissions", ["kb_id"])
    op.create_index("ix_kb_user_permissions_user_id", "kb_user_permissions", ["user_id"])


def downgrade():
    op.drop_index("ix_kb_user_permissions_user_id", table_name="kb_user_permissions")
    op.drop_index("ix_kb_user_permissions_kb_id", table_name="kb_user_permissions")
    op.drop_table("kb_user_permissions")
    op.drop_index("ix_knowledge_bases_user_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
