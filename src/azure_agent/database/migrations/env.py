from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from azure_agent.config import require_env


config = context.config
target_metadata = None

# Azure Database for PostgreSQL
postgres_url = require_env("POSTGRES_WEB_CONN_STRING")
database_url = postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)

# Run migrations (offline)
if context.is_offline_mode():
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
# Run migrations (online)
else:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()
