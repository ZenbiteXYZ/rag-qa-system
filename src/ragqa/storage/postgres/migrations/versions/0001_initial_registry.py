"""Initial registry: documents, versions, indexes, builds and jobs.

Revision ID: 0001
Revises:
Create Date: 2026-10-20
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATUSES = "status IN ('queued', 'parsing', 'chunking', 'embedding', 'ready', 'failed', 'dead')"
EMPTY_JSON = sa.text("'{}'::jsonb")


def _timestamp(name: str, *, default: bool = True) -> sa.Column[datetime]:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.text("now()") if default else None,
        nullable=not default,
    )


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        # The foreign key to versions is added below, once that table exists.
        sa.Column("latest_version_id", sa.UUID(), nullable=True),
        _timestamp("created_at"),
        _timestamp("deleted_at", default=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_table(
        "indexes",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("collection", sa.Text(), nullable=False),
        sa.Column("dense_model", sa.Text(), nullable=False),
        sa.Column("dense_vector_size", sa.Integer(), nullable=False),
        sa.Column("dense_distance", sa.Text(), server_default="cosine", nullable=False),
        sa.Column("sparse_model", sa.Text(), nullable=True),
        sa.Column("chunking_strategy", sa.Text(), nullable=False),
        sa.Column("chunking_params", postgresql.JSONB(), server_default=EMPTY_JSON, nullable=False),
        _timestamp("created_at"),
        sa.CheckConstraint(
            "dense_distance IN ('cosine', 'dot', 'euclid')",
            name=op.f("ck_indexes_dense_distance_known"),
        ),
        sa.CheckConstraint(
            "dense_vector_size > 0", name=op.f("ck_indexes_dense_vector_size_positive")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_indexes")),
        sa.UniqueConstraint("collection", name=op.f("uq_indexes_collection")),
        sa.UniqueConstraint("name", name=op.f("uq_indexes_name")),
    )
    op.create_table(
        "versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("source_format", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("raw_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=EMPTY_JSON, nullable=False),
        _timestamp("uploaded_at"),
        sa.CheckConstraint("number >= 1", name=op.f("ck_versions_number_positive")),
        sa.CheckConstraint("size_bytes >= 0", name=op.f("ck_versions_size_not_negative")),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_versions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_versions")),
        sa.UniqueConstraint("document_id", "number", name=op.f("uq_versions_document_id_number")),
    )
    op.create_index(op.f("ix_versions_document_id"), "versions", ["document_id"])
    op.create_foreign_key(
        op.f("fk_documents_latest_version_id_versions"),
        "documents",
        "versions",
        ["latest_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("index_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), server_default="queued", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        _timestamp("created_at"),
        _timestamp("updated_at"),
        sa.CheckConstraint(STATUSES, name=op.f("ck_jobs_status_known")),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_jobs_attempts_not_negative")),
        sa.ForeignKeyConstraint(
            ["index_id"], ["indexes.id"], name=op.f("fk_jobs_index_id_indexes"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["versions.id"],
            name=op.f("fk_jobs_version_id_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index(op.f("ix_jobs_status_updated_at"), "jobs", ["status", "updated_at"])
    op.create_index(op.f("ix_jobs_version_id"), "jobs", ["version_id"])
    op.create_table(
        "version_index_builds",
        sa.Column("version_id", sa.UUID(), nullable=False),
        sa.Column("index_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        _timestamp("built_at", default=False),
        sa.CheckConstraint(STATUSES, name=op.f("ck_version_index_builds_status_known")),
        sa.ForeignKeyConstraint(
            ["index_id"],
            ["indexes.id"],
            name=op.f("fk_version_index_builds_index_id_indexes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["versions.id"],
            name=op.f("fk_version_index_builds_version_id_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("version_id", "index_id", name=op.f("pk_version_index_builds")),
    )


def downgrade() -> None:
    op.drop_table("version_index_builds")
    op.drop_table("jobs")
    op.drop_constraint(
        op.f("fk_documents_latest_version_id_versions"), "documents", type_="foreignkey"
    )
    op.drop_table("versions")
    op.drop_table("indexes")
    op.drop_table("documents")
