from __future__ import annotations

import os
from dataclasses import dataclass


def _normalize_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return _normalize_env_value(value)


@dataclass(frozen=True, slots=True)
class RedisStreamSettings:
    host: str
    username: str
    access_key: str
    port: int


@dataclass(frozen=True, slots=True)
class ApiSettings:
    blob_connection_string: str
    postgres_web_conn_string: str
    redis_stream: RedisStreamSettings


@dataclass(frozen=True, slots=True)
class GraphSettings:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_main_model: str
    azure_openai_main_model_timeout: int
    azure_openai_small_model: str
    azure_openai_small_model_timeout: int
    azure_openai_embedding_model: str
    azure_openai_embedding_dims: int
    azure_ai_search_endpoint: str
    azure_ai_search_api_key: str
    azure_ai_search_index_name: str
    azure_ai_search_semantic_config: str
    azure_ai_search_api_version: str
    azure_ai_search_top_k: int
    blob_connection_string: str
    redis_host: str
    redis_username: str
    redis_access_key: str
    redis_port: int
    redis_db: int
    postgres_conn_string: str
    postgres_web_conn_string: str
    azure_ai_content_safety_endpoint: str
    azure_ai_content_safety_api_key: str
    azure_dynamic_sessions_python_pool_endpoint: str
    azure_dynamic_sessions_bash_pool_endpoint: str
    langfuse_base_url: str
    langfuse_public_key: str
    langfuse_secret_key: str


def load_redis_stream_settings() -> RedisStreamSettings:
    return RedisStreamSettings(
        host=require_env("REDIS_STREAM_HOST"),
        username=require_env("REDIS_STREAM_USERNAME"),
        access_key=require_env("REDIS_STREAM_ACCESS_KEY"),
        port=int(os.getenv("REDIS_STREAM_PORT") or 10000),
    )


def load_api_settings() -> ApiSettings:
    return ApiSettings(
        blob_connection_string=require_env("BLOB_CONNECTION_STRING"),
        postgres_web_conn_string=require_env("POSTGRES_WEB_CONN_STRING"),
        redis_stream=load_redis_stream_settings(),
    )


def load_graph_settings() -> GraphSettings:
    return GraphSettings(
        azure_openai_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=require_env("AZURE_OPENAI_API_KEY"),
        azure_openai_api_version=require_env("AZURE_OPENAI_API_VERSION"),
        azure_openai_main_model=require_env("AZURE_OPENAI_MAIN_MODEL"),
        azure_openai_main_model_timeout=int(
            require_env("AZURE_OPENAI_MAIN_MODEL_TIMEOUT")
        ),
        azure_openai_small_model=require_env("AZURE_OPENAI_SMALL_MODEL"),
        azure_openai_small_model_timeout=int(
            require_env("AZURE_OPENAI_SMALL_MODEL_TIMEOUT")
        ),
        azure_openai_embedding_model=require_env("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_openai_embedding_dims=int(require_env("AZURE_OPENAI_EMBEDDING_DIMS")),
        azure_ai_search_endpoint=require_env("AZURE_AI_SEARCH_ENDPOINT"),
        azure_ai_search_api_key=require_env("AZURE_AI_SEARCH_API_KEY"),
        azure_ai_search_index_name=require_env("AZURE_AI_SEARCH_INDEX_NAME"),
        azure_ai_search_semantic_config=require_env(
            "AZURE_AI_SEARCH_SEMANTIC_CONFIG"
        ),
        azure_ai_search_api_version=require_env("AZURE_AI_SEARCH_API_VERSION"),
        azure_ai_search_top_k=int(require_env("AZURE_AI_SEARCH_TOP_K")),
        blob_connection_string=require_env("BLOB_CONNECTION_STRING"),
        redis_host=require_env("REDIS_HOST"),
        redis_username=require_env("REDIS_USERNAME"),
        redis_access_key=require_env("REDIS_ACCESS_KEY"),
        redis_port=int(require_env("REDIS_PORT")),
        redis_db=int(require_env("REDIS_DB")),
        postgres_conn_string=require_env("POSTGRES_CONN_STRING"),
        postgres_web_conn_string=require_env("POSTGRES_WEB_CONN_STRING"),
        azure_ai_content_safety_endpoint=require_env(
            "AZURE_AI_CONTENT_SAFETY_ENDPOINT"
        ),
        azure_ai_content_safety_api_key=require_env("AZURE_AI_CONTENT_SAFETY_API_KEY"),
        azure_dynamic_sessions_python_pool_endpoint=require_env(
            "AZURE_DYNAMIC_SESSIONS_PYTHON_POOL_ENDPOINT"
        ),
        azure_dynamic_sessions_bash_pool_endpoint=require_env(
            "AZURE_DYNAMIC_SESSIONS_BASH_POOL_ENDPOINT"
        ),
        langfuse_base_url=require_env("LANGFUSE_BASE_URL"),
        langfuse_public_key=require_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=require_env("LANGFUSE_SECRET_KEY"),
    )
