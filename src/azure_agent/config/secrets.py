from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields

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
    # Tavily
    TAVILY_API_KEY: str
    # Tiktoken
    TIKTOKEN_ENCODER: str


async def load_app_secrets(secret_client: SecretClient) -> AppSecrets:
    field_names = [item.name for item in fields(AppSecrets)]
    bundles = await asyncio.gather(
        *(secret_client.get_secret(name.replace("_", "-")) for name in field_names)
    )
    values = {
        name: bundle.value
        for name, bundle in zip(field_names, bundles)
    }
    return AppSecrets(**values)
