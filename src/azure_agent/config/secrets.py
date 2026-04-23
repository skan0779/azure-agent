from __future__ import annotations

import asyncio
from dataclasses import MISSING, dataclass, fields

from azure.core.exceptions import ResourceNotFoundError
from azure.keyvault.secrets.aio import SecretClient


@dataclass(frozen=True, slots=True)
class AppSecrets:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_MAIN_MODEL: str
    AZURE_OPENAI_MAIN_MODEL_TIMEOUT: str
    AZURE_OPENAI_SMALL_MODEL: str
    AZURE_OPENAI_SMALL_MODEL_TIMEOUT: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    AZURE_OPENAI_EMBEDDING_DIMS: str
    # Azure AI Search
    AZURE_AI_SEARCH_ENDPOINT: str
    AZURE_AI_SEARCH_API_KEY: str
    AZURE_AI_SEARCH_INDEX_NAME: str
    AZURE_AI_SEARCH_SEMANTIC_CONFIG: str
    AZURE_AI_SEARCH_API_VERSION: str
    AZURE_AI_SEARCH_TOP_K: str
    # Azure Blob Storage
    BLOB_CONTAINER_NAME: str
    BLOB_CONNECTION_STRING: str
    # Azure Managed Redis
    REDIS_HOST: str
    REDIS_USERNAME: str
    REDIS_ACCESS_KEY: str
    REDIS_PORT: str
    REDIS_DB: str
    # Azure Database for PostgreSQL
    POSTGRES_CONN_STRING: str
    # Tiktoken
    TIKTOKEN_ENCODER: str
    # Azure AI Content Safety
    AZURE_AI_CONTENT_SAFETY_ENDPOINT: str
    AZURE_AI_CONTENT_SAFETY_API_KEY: str
    # Azure Container Apps Dynamic Sessions
    AZURE_DYNAMIC_SESSIONS_POOL_ENDPOINT: str


async def load_app_secrets(secret_client: SecretClient) -> AppSecrets:
    async def get_secret_value(field_name, default=MISSING):
        secret_name = field_name.replace("_", "-")
        try:
            bundle = await secret_client.get_secret(secret_name)
            return bundle.value
        except ResourceNotFoundError:
            if default is not MISSING:
                return default
            raise RuntimeError(f"Missing Key Vault secret: {secret_name}")

    field_defs = fields(AppSecrets)
    values_list = await asyncio.gather(
        *(
            get_secret_value(
                field_def.name,
                default=field_def.default,
            )
            for field_def in field_defs
        )
    )
    values = {
        field_def.name: value
        for field_def, value in zip(field_defs, values_list)
    }
    return AppSecrets(**values)
