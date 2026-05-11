from __future__ import annotations

import os

from alembic import context
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from sqlalchemy import engine_from_config, pool


config = context.config
target_metadata = None

# Azure Key Vault
vault_url = os.getenv("KEY_VAULT_URL")
if not vault_url:
    raise RuntimeError("KEY_VAULT_URL is required for database migrations.")
credential = DefaultAzureCredential()
secret_client = SecretClient(
    vault_url=vault_url,
    credential=credential
)

# Azure Database for PostgreSQL
try:
    postgres_url = (secret_client.get_secret("POSTGRES-WEB-CONN-STRING").value or "").strip()
finally:
    secret_client.close()
    credential.close()
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
