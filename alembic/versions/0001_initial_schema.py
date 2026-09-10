"""Initial schema: corpus, AI artefacts, decisions, jobs, cost ledger

Revision ID: 0001_initial
Revises: 
Create Date: 2026-08-31 21:43:08.081461
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('ai_calls',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('operation', sa.String(length=30), nullable=False),
    sa.Column('provider', sa.String(length=40), nullable=False),
    sa.Column('model', sa.String(length=120), nullable=False),
    sa.Column('subject_ref', sa.String(length=120), nullable=False),
    sa.Column('job_id', sa.String(length=32), nullable=True),
    sa.Column('input_tokens', sa.Integer(), nullable=False),
    sa.Column('output_tokens', sa.Integer(), nullable=False),
    sa.Column('cost_usd', sa.Float(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('ai_calls', schema=None) as batch_op:
        batch_op.create_index('ix_ai_calls_job', ['job_id'], unique=False)
        batch_op.create_index('ix_ai_calls_tenant_created', ['tenant_id', 'created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_ai_calls_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('embeddings',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('owner_type', sa.String(length=20), nullable=False),
    sa.Column('owner_id', sa.String(length=32), nullable=False),
    sa.Column('model', sa.String(length=120), nullable=False),
    sa.Column('dim', sa.Integer(), nullable=False),
    sa.Column('vector', sa.JSON(), nullable=False),
    sa.Column('source_text', sa.Text(), nullable=False),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'owner_type', 'owner_id', 'model', name='uq_embeddings_owner_model')
    )
    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_embeddings_tenant_id'), ['tenant_id'], unique=False)
        batch_op.create_index('ix_embeddings_tenant_owner_model', ['tenant_id', 'owner_type', 'model'], unique=False)

    op.create_table('images',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('filename', sa.String(length=512), nullable=False),
    sa.Column('path', sa.String(length=1024), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('bytes', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('subject', sa.String(length=200), nullable=True),
    sa.Column('category', sa.String(length=64), nullable=True),
    sa.Column('caption', sa.Text(), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('attributes', sa.JSON(), nullable=True),
    sa.Column('needs_review', sa.Boolean(), nullable=False),
    sa.Column('review_reason', sa.String(length=300), nullable=True),
    sa.Column('raw_response', sa.JSON(), nullable=True),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('vision_model', sa.String(length=120), nullable=True),
    sa.Column('tagged_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'filename', name='uq_images_tenant_filename'),
    sa.UniqueConstraint('tenant_id', 'sha256', name='uq_images_tenant_sha256')
    )
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.create_index('ix_images_tenant_category', ['tenant_id', 'category'], unique=False)
        batch_op.create_index(batch_op.f('ix_images_tenant_id'), ['tenant_id'], unique=False)
        batch_op.create_index('ix_images_tenant_review', ['tenant_id', 'needs_review'], unique=False)
        batch_op.create_index('ix_images_tenant_status', ['tenant_id', 'status'], unique=False)

    op.create_table('jobs',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('kind', sa.String(length=60), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('dedupe_key', sa.String(length=200), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('max_attempts', sa.Integer(), nullable=False),
    sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('total_items', sa.Integer(), nullable=False),
    sa.Column('processed_items', sa.Integer(), nullable=False),
    sa.Column('failed_items', sa.Integer(), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'dedupe_key', name='uq_jobs_dedupe')
    )
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.create_index('ix_jobs_status_scheduled', ['status', 'scheduled_at'], unique=False)
        batch_op.create_index('ix_jobs_tenant_created', ['tenant_id', 'created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_jobs_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('posts',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('slug', sa.String(length=200), nullable=False),
    sa.Column('title', sa.String(length=300), nullable=False),
    sa.Column('body', sa.Text(), nullable=False),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'slug', name='uq_posts_tenant_slug')
    )
    with op.batch_alter_table('posts', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_posts_tenant_id'), ['tenant_id'], unique=False)

    op.create_table('tenants',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('image_tags',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('image_id', sa.String(length=32), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('value', sa.String(length=120), nullable=False),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('image_id', 'kind', 'value', name='uq_image_tag')
    )
    with op.batch_alter_table('image_tags', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_image_tags_image_id'), ['image_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_image_tags_tenant_id'), ['tenant_id'], unique=False)
        batch_op.create_index('ix_image_tags_tenant_value', ['tenant_id', 'value'], unique=False)

    op.create_table('suggestions',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('post_id', sa.String(length=32), nullable=False),
    sa.Column('image_id', sa.String(length=32), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=False),
    sa.Column('similarity', sa.Float(), nullable=False),
    sa.Column('verdict', sa.String(length=20), nullable=False),
    sa.Column('reasons', sa.JSON(), nullable=False),
    sa.Column('explanation', sa.JSON(), nullable=False),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['post_id'], ['posts.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'post_id', 'image_id', name='uq_suggestions_post_image')
    )
    with op.batch_alter_table('suggestions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_suggestions_tenant_id'), ['tenant_id'], unique=False)
        batch_op.create_index('ix_suggestions_tenant_post_rank', ['tenant_id', 'post_id', 'rank'], unique=False)
        batch_op.create_index('ix_suggestions_tenant_verdict', ['tenant_id', 'verdict'], unique=False)

    op.create_table('reviews',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('suggestion_id', sa.String(length=32), nullable=False),
    sa.Column('decision', sa.String(length=20), nullable=False),
    sa.Column('reviewer', sa.String(length=120), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('idempotency_key', sa.String(length=120), nullable=False),
    sa.Column('tenant_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['suggestion_id'], ['suggestions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'idempotency_key', name='uq_reviews_idempotency')
    )
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_reviews_tenant_id'), ['tenant_id'], unique=False)
        batch_op.create_index('ix_reviews_tenant_suggestion', ['tenant_id', 'suggestion_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.drop_index('ix_reviews_tenant_suggestion')
        batch_op.drop_index(batch_op.f('ix_reviews_tenant_id'))

    op.drop_table('reviews')
    with op.batch_alter_table('suggestions', schema=None) as batch_op:
        batch_op.drop_index('ix_suggestions_tenant_verdict')
        batch_op.drop_index('ix_suggestions_tenant_post_rank')
        batch_op.drop_index(batch_op.f('ix_suggestions_tenant_id'))

    op.drop_table('suggestions')
    with op.batch_alter_table('image_tags', schema=None) as batch_op:
        batch_op.drop_index('ix_image_tags_tenant_value')
        batch_op.drop_index(batch_op.f('ix_image_tags_tenant_id'))
        batch_op.drop_index(batch_op.f('ix_image_tags_image_id'))

    op.drop_table('image_tags')
    op.drop_table('tenants')
    with op.batch_alter_table('posts', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_posts_tenant_id'))

    op.drop_table('posts')
    with op.batch_alter_table('jobs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_jobs_tenant_id'))
        batch_op.drop_index('ix_jobs_tenant_created')
        batch_op.drop_index('ix_jobs_status_scheduled')

    op.drop_table('jobs')
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.drop_index('ix_images_tenant_status')
        batch_op.drop_index('ix_images_tenant_review')
        batch_op.drop_index(batch_op.f('ix_images_tenant_id'))
        batch_op.drop_index('ix_images_tenant_category')

    op.drop_table('images')
    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.drop_index('ix_embeddings_tenant_owner_model')
        batch_op.drop_index(batch_op.f('ix_embeddings_tenant_id'))

    op.drop_table('embeddings')
    with op.batch_alter_table('ai_calls', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_ai_calls_tenant_id'))
        batch_op.drop_index('ix_ai_calls_tenant_created')
        batch_op.drop_index('ix_ai_calls_job')

    op.drop_table('ai_calls')
