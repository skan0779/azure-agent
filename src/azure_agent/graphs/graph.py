import asyncio, os, yaml, logging, inspect, json
from pathlib import Path
from collections.abc import Awaitable, Callable

from azure.storage.blob.aio import ContainerClient
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from redis.asyncio import Redis

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    PIIMiddleware,
)

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

from langchain_tavily import TavilySearch

from langchain_community.vectorstores.azuresearch import AzureSearch

from langmem import create_manage_memory_tool, create_search_memory_tool

from azure_agent.middlewares.stream import (
    event_stream_before_agent,
    event_stream_before_model,
)
from azure_agent.infra.key_vault import create_async_secret_client
from azure_agent.graphs.schema import AgentState, UserProfile
from azure_agent.tools.azure_ai_search import create_azure_ai_search_tool

logger = logging.getLogger(__name__)


class LangGraphProcess:
    """
    LangGraphProcess Application Configuration and Runtime Resource Manager that Performs:
        - Load Azure Key Vault Secrets
        - Create Azure OpenAI Models (Main, Small, Embedding)
        - Create Azure AI Search Client
        - Create Redis Client
        - Create Checkpointer (Shallow Redis)
        - Create Store (Postgres)
        - Build Agent runnable
    Args:
        vault_url: Optional Azure Key Vault URL override
    """
    def __init__(self, vault_url: str | None = None) -> None:
        """
        Initialize lightweight application state.
        """
        self.vault_url = vault_url
        self.secret_client: SecretClient | None = None
        self.secret_credential: DefaultAzureCredential | None = None
        self.prompt_cache: dict[str, str] = {}

        # Secrets / config
        self.AZURE_OPENAI_ENDPOINT: str | None = None
        self.AZURE_OPENAI_API_KEY: str | None = None
        self.AZURE_OPENAI_API_VERSION: str | None = None
        self.AZURE_OPENAI_MAIN_MODEL: str | None = None
        self.AZURE_OPENAI_MAIN_MODEL_TIMEOUT: str | None = None
        self.AZURE_OPENAI_SMALL_MODEL: str | None = None
        self.AZURE_OPENAI_SMALL_MODEL_TIMEOUT: str | None = None
        self.AZURE_OPENAI_EMBEDDING_MODEL: str | None = None
        self.AZURE_OPENAI_EMBEDDING_DIMS: str | None = None
        self.AZURE_AI_SEARCH_ENDPOINT: str | None = None
        self.AZURE_AI_SEARCH_API_KEY: str | None = None
        self.AZURE_AI_SEARCH_INDEX_NAME: str | None = None
        self.AZURE_AI_SEARCH_SEMANTIC_CONFIG: str | None = None
        self.AZURE_AI_SEARCH_API_VERSION: str | None = None
        self.AZURE_AI_SEARCH_TOP_K: str | None = None
        self.BLOB_CONTAINER_NAME: str | None = None
        self.BLOB_CONNECTION_STRING: str | None = None
        self.REDIS_HOST: str | None = None
        self.REDIS_USERNAME: str | None = None
        self.REDIS_ACCESS_KEY: str | None = None
        self.REDIS_PORT: str | None = None
        self.REDIS_DB: str | None = None
        self.POSTGRES_CONN_STRING: str | None = None
        self.TIKTOKEN_ENCODER: str | None = None

        # Runtime clients / models
        self.BLOB_CONTAINER_CLIENT: ContainerClient | None = None
        self.main_model = None
        self.small_model = None
        self.embedding_model = None
        self.azure_search = None
        self.redis_client = None
        self.memory = None
        self.store = None
        self._store_cm = None
        self.graph = None

        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(__file__).resolve().parent.parent / "encoder")

    async def setup(self):
        """
        Initalize Application Runtime Resource that Performs:
            Load Azure Key Vault Secrets
            Create Azure OpenAI Models (Main, Small, Embedding)
            Create Azure AI Search Client
            Create Blob Client
            Create Redis Client
            Create Checkpointer (Shallow Redis): ttl 1 day
            Create  Store (Postgres): ttl 30 days
            Load Prompt Cache
            Build Agent runnable
        """
        await self._load_secrets()
        self._load_instance()

        # Redis Client
        self.redis_client = Redis(
            host=self.REDIS_HOST,
            port=int(self.REDIS_PORT or 10000),
            username=self.REDIS_USERNAME,
            password=self.REDIS_ACCESS_KEY,
            db=int(self.REDIS_DB or 0),
            decode_responses=False,
            ssl=True,
            socket_connect_timeout=5,
            socket_timeout=20,
            retry_on_timeout=True
        )

        # Checkpointer
        self.memory = AsyncShallowRedisSaver(
            redis_client=self.redis_client,
            ttl={
                "default_ttl": 60 * 60 * 24 * 1, # 1 day
                "refresh_on_read": True,
            },
        )
        await getattr(self.memory, "asetup", lambda: None)()

        # Store 
        store_cm = AsyncPostgresStore.from_conn_string(
            conn_string=self.POSTGRES_CONN_STRING,
            pool_config=PoolConfig(
                min_size=1,
                max_size=5,
                max_lifetime=60 * 30,
                max_idle=60 * 5,
                kwargs={
                    "connect_timeout": 5,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3,
                },
            ),
            ttl={
                "default_ttl": 60 * 60 * 24 * 30, # 30 days
                "refresh_on_read": True, 
            },
            index={
                "dims": int(self.AZURE_OPENAI_EMBEDDING_DIMS),
                "embed": self.embedding_model,
            }
        )
        
        self._store_cm = store_cm
        self.store = await self._store_cm.__aenter__()

        setup = getattr(self.store, "asetup", None) or getattr(self.store, "setup", None)
        if callable(setup):
            maybe = setup()
            if inspect.isawaitable(maybe):
                await maybe

        await self._load_prompts(["example.yaml"])

        # Build Graph
        self.graph = self._build_graph(
            checkpointer=self.memory,
            store=self.store,
        )

    async def _load_secrets(self) -> None:
        self.secret_client, self.secret_credential = create_async_secret_client(
            self.vault_url
        )

        secret_names = {
            "AZURE_OPENAI_ENDPOINT": "AZURE-OPENAI-ENDPOINT",
            "AZURE_OPENAI_API_KEY": "AZURE-OPENAI-API-KEY",
            "AZURE_OPENAI_API_VERSION": "AZURE-OPENAI-API-VERSION",
            "AZURE_OPENAI_MAIN_MODEL": "AZURE-OPENAI-MAIN-MODEL",
            "AZURE_OPENAI_MAIN_MODEL_TIMEOUT": "AZURE-OPENAI-MAIN-MODEL-TIMEOUT",
            "AZURE_OPENAI_SMALL_MODEL": "AZURE-OPENAI-SMALL-MODEL",
            "AZURE_OPENAI_SMALL_MODEL_TIMEOUT": "AZURE-OPENAI-SMALL-MODEL-TIMEOUT",
            "AZURE_OPENAI_EMBEDDING_MODEL": "AZURE-OPENAI-EMBEDDING-MODEL",
            "AZURE_OPENAI_EMBEDDING_DIMS": "AZURE-OPENAI-EMBEDDING-DIMS",
            "AZURE_AI_SEARCH_ENDPOINT": "AZURE-AI-SEARCH-ENDPOINT",
            "AZURE_AI_SEARCH_API_KEY": "AZURE-AI-SEARCH-API-KEY",
            "AZURE_AI_SEARCH_INDEX_NAME": "AZURE-AI-SEARCH-INDEX-NAME",
            "AZURE_AI_SEARCH_SEMANTIC_CONFIG": "AZURE-AI-SEARCH-SEMANTIC-CONFIG",
            "AZURE_AI_SEARCH_API_VERSION": "AZURE-AI-SEARCH-API-VERSION",
            "AZURE_AI_SEARCH_TOP_K": "AZURE-AI-SEARCH-TOP-K",
            "BLOB_CONTAINER_NAME": "BLOB-CONTAINER-NAME",
            "BLOB_CONNECTION_STRING": "BLOB-CONNECTION-STRING",
            "REDIS_HOST": "REDIS-HOST",
            "REDIS_USERNAME": "REDIS-USERNAME",
            "REDIS_ACCESS_KEY": "REDIS-ACCESS-KEY",
            "REDIS_PORT": "REDIS-PORT",
            "REDIS_DB": "REDIS-DB",
            "POSTGRES_CONN_STRING": "POSTGRES-CONN-STRING",
            "TAVILY_API_KEY": "TAVILY-API-KEY",
            "TIKTOKEN_ENCODER": "TIKTOKEN-ENCODER",
        }

        assert self.secret_client is not None
        bundles = await asyncio.gather(
            *(self.secret_client.get_secret(name) for name in secret_names.values())
        )
        for attr, bundle in zip(secret_names.keys(), bundles):
            setattr(self, attr, bundle.value)

        os.environ["TAVILY_API_KEY"] = str(getattr(self, "TAVILY_API_KEY", ""))

    def _load_instance(self) -> None:
        self.BLOB_CONTAINER_CLIENT = ContainerClient.from_connection_string(
            conn_str=str(self.BLOB_CONNECTION_STRING),
            container_name=str(self.BLOB_CONTAINER_NAME),
        )

        self.main_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_MAIN_MODEL,
            tiktoken_model_name=self.AZURE_OPENAI_MAIN_MODEL,
            model=self.AZURE_OPENAI_MAIN_MODEL,
            stream_usage=True,
            request_timeout=int(self.AZURE_OPENAI_MAIN_MODEL_TIMEOUT),
        )

        self.small_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_SMALL_MODEL,
            tiktoken_model_name=self.AZURE_OPENAI_SMALL_MODEL,
            model=self.AZURE_OPENAI_SMALL_MODEL,
            streaming=True,
            stream_usage=False,
            request_timeout=int(self.AZURE_OPENAI_SMALL_MODEL_TIMEOUT),
        )

        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_EMBEDDING_MODEL,
            model=self.AZURE_OPENAI_EMBEDDING_MODEL,
        )

        self.azure_search = AzureSearch(
            azure_search_endpoint=str(self.AZURE_AI_SEARCH_ENDPOINT),
            azure_search_key=str(self.AZURE_AI_SEARCH_API_KEY),
            index_name=str(self.AZURE_AI_SEARCH_INDEX_NAME),
            embedding_function=self.embedding_model,
            search_type="semantic_hybrid",
            semantic_configuration_name=str(self.AZURE_AI_SEARCH_SEMANTIC_CONFIG),
            vector_search_dimensions=int(self.AZURE_OPENAI_EMBEDDING_DIMS),
            additional_search_client_options={
                "api_version": str(self.AZURE_AI_SEARCH_API_VERSION),
            },
        )

    async def _load_prompts(self, file_names: list[str]) -> None:
        for file_name in file_names:
            if file_name in self.prompt_cache:
                continue
            # Download from Blob Storage
            try:
                if self.BLOB_CONTAINER_CLIENT is None:
                    raise RuntimeError("Blob container client is not initialized")
                downloader = await self.BLOB_CONTAINER_CLIENT.download_blob(file_name)
                raw = (await downloader.readall()).decode("utf-8")

            # Load from Local Repository
            except Exception as exc:
                logger.warning(
                    "[graph.py] Failed to load system prompt from blob storage (%s): %s",
                    file_name,
                    exc,
                )
                prompt_path = Path(__file__).resolve().parents[1] / "prompts" / file_name
                try:
                    raw = await asyncio.to_thread(prompt_path.read_text, encoding="utf-8")
                except FileNotFoundError:
                    logger.error(
                        "[graph.py] Failed to load system prompt from local file: %s",
                        prompt_path,
                    )
                    raise

            data = yaml.safe_load(raw)
            self.prompt_cache[file_name] = str(data["system"]).strip()

    async def close(self) -> None:
        """
        Cleanup Application Runtime Resources that Performs:
            - Stop Store Sweeper
            - Cleanup Store
            - Cleanup Store context manager
            - Cleanup Checkpointer
            - Cleanup Redis Client, Connection Pool
            - Cleanup Azure AI Search Client
        """
        # Cleanup Blob Client
        blob_container_client = getattr(self, "BLOB_CONTAINER_CLIENT", None)
        if blob_container_client is not None:
            try:
                close = getattr(blob_container_client, "close", None)
                if callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close blob container client: %s", exc)
            self.BLOB_CONTAINER_CLIENT = None

        # Cleanup Store
        store = getattr(self, "store", None)
        if store is not None:
            # Stop Sweeper
            try:
                stop = getattr(store, "stop_ttl_sweeper", None)
                if callable(stop):
                    try:
                        maybe = stop(timeout=5)
                        if inspect.isawaitable(maybe):
                            await maybe
                    except TypeError:
                        maybe = stop()
                        if inspect.isawaitable(maybe):
                            await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to stop store sweeper: %s", exc)

            # Close Store
            try:
                aclose = getattr(store, "aclose", None)
                close = getattr(store, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close store: %s", exc)
            
            # Remove Reference
            self.store = None
        
        # Cleanup Store context manager
        store_cm = getattr(self, "_store_cm", None)
        if store_cm is not None:
            try:
                await store_cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("[graph.py] Failed to exit store context manager: %s", exc)
            self._store_cm = None
        
        # Cleanup Checkpointer
        memory = getattr(self, "memory", None)
        if memory is not None:
            try:
                aclose = getattr(memory, "aclose", None)
                close = getattr(memory, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close checkpointer memory: %s", exc)

            # Remove Reference
            self.memory = None

        # Cleanup Redis Client
        redis_client = getattr(self, "redis_client", None)
        if redis_client is not None:

            # Cleanup Redis Client
            try:
                aclose = getattr(redis_client, "aclose", None)
                close = getattr(redis_client, "close", None)
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close redis client: %s", exc)
            
            # Cleanup Redis Connection Pool
            try:
                pool = getattr(redis_client, "connection_pool", None)
                if pool is not None:
                    disconnect = getattr(pool, "disconnect", None)
                    if callable(disconnect):
                        maybe = disconnect()
                        if inspect.isawaitable(maybe):
                            await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to disconnect redis connection pool: %s", exc)

            # Remove Reference
            self.redis_client = None

        # Cleanup Azure AI Search clients
        azure_search = getattr(self, "azure_search", None)
        if azure_search is not None:
            try:
                async_client = getattr(azure_search, "async_client", None)
                if async_client is not None:
                    close = getattr(async_client, "close", None)
                    if callable(close):
                        maybe = close()
                        if inspect.isawaitable(maybe):
                            await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close Azure AI Search async client: %s", exc)

            try:
                client = getattr(azure_search, "client", None)
                if client is not None:
                    close = getattr(client, "close", None)
                    if callable(close):
                        maybe = close()
                        if inspect.isawaitable(maybe):
                            await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close Azure AI Search client: %s", exc)

            self.azure_search = None

        # Cleanup Key Vault Client
        secret_client = getattr(self, "secret_client", None)
        if secret_client is not None:
            try:
                close = getattr(secret_client, "close", None)
                if callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close Key Vault client: %s", exc)
            self.secret_client = None

        secret_credential = getattr(self, "secret_credential", None)
        if secret_credential is not None:
            try:
                close = getattr(secret_credential, "close", None)
                if callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close Key Vault credential: %s", exc)
            self.secret_credential = None

    def _build_graph(self, checkpointer=None, store=None):
        """
        LangGraphProcess builder that returns the agent runnable:
            Tools:
                azure_ai_search_tool: RAG tool
            Agent:
                main_agent: main conversational react agent
        
        Args:
            checkpointer: checkpoint saver Instance
            store: data store Instance
        Returns:
            Runnable: Agent runnable
        """
        # Create Manage Memory Tool
        manage_memory = create_manage_memory_tool(
            namespace=("memories", "{user_id}", "profile"),
            schema=UserProfile,
            actions_permitted=("create", "update", "delete"),
            name="manage_memory",
        )

        # Create Search Memory Tool
        search_memory = create_search_memory_tool(
            namespace=("memories", "{user_id}"),
            name="search_memory",
        )

        # Create Azure AI Search Tool
        azure_ai_search = create_azure_ai_search_tool(
            azure_ai_search=self.azure_search,
            top_k=int(self.AZURE_AI_SEARCH_TOP_K),
        )

        # Create Tavily Search Tool
        tavily_search = TavilySearch(
            max_results=3, 
            topic="general"
        )
        
        # Create Main Agent
        main_agent = create_agent(
            model=self.main_model,
            tools=[
                manage_memory,          # LangMem Tool
                search_memory,          # LangMem Tool
                azure_ai_search,        # RAG Tool
                tavily_search,          # Web Search Tool
            ],
            system_prompt=self.prompt_cache["example.yaml"],
            middleware=[
                # Model Middleware
                ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),

                # Tool Middleware
                ToolCallLimitMiddleware(run_limit=5, exit_behavior="continue"),

                # Message Middleware
                SummarizationMiddleware(
                    model=self.small_model,
                    trigger=[("tokens", 20000)],
                    keep=("messages", 20),
                    token_counter=self.small_model.get_num_tokens_from_messages,
                ),

                # PII Middleware
                PIIMiddleware("email", strategy="mask"),
                PIIMiddleware("credit_card", strategy="mask"),

                # Custom Middleware
                event_stream_before_agent,
                event_stream_before_model,
            ],
            state_schema=AgentState,
            checkpointer=checkpointer,
            store=store,
            name="main_agent",
        )

        return main_agent

    async def main(
        self,
        thread_id: str,
        user_id: str,
        user_query: str,
    ):
        """
        LangGraphProcess Main Function

        Args:
            thread_id: Thread ID (Session ID)
            user_id: User ID
            user_query: User Query
        Yields:
            Streamed Events (Messages, Custom Events)
        """
        
        # Logging
        logger.info("[graph.py] LangGraphProcess Request : thread_id=%s, user_id=%s, user_query=%s", thread_id, user_id, user_query)

        # Input Values
        inputs = {
            "messages": [HumanMessage(content=user_query)],
            "thread_id": thread_id,
            "user_id": user_id,
            "user_query": user_query,
        }

        # Runnable Config
        config = RunnableConfig(
            recursion_limit=50,
            configurable={
                "thread_id": thread_id,
                "user_id": user_id,
            },
        )

        # Stream Processing
        stream = self.graph.astream(
            inputs, 
            config,
            subgraphs=True,
            stream_mode=["messages", "updates", "custom"],
        )
        try:
            async for event in stream:
                namespace = None
                mode = None
                payload = None
                data = event

                # Parse Payload (namespace, data)
                if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], tuple):
                    namespace, data = data

                # Parse Payload (mode, payload)
                if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
                    mode, payload = data
                elif isinstance(data, tuple) and len(data) == 3 and isinstance(data[1], str):
                    namespace, mode, payload = data
                elif isinstance(data, tuple) and len(data) == 2:
                    mode, payload = "messages", data
                elif isinstance(data, dict):
                    mode, payload = "updates", data
                else:
                    mode, payload = "custom", data

                # Stream Messages
                if mode == "messages":

                    STREAM_NODES = {"model"}

                    # Parse Payload
                    msg, metadata = payload, None
                    if isinstance(payload, tuple) and len(payload) == 2:
                        msg, metadata = payload

                    # Filter AIMessageChunk
                    if not isinstance(msg, AIMessageChunk):
                        continue

                    # Parse Node
                    node = None
                    if isinstance(metadata, dict):
                        node = metadata.get("langgraph_node")
                    
                    # Filter Node
                    if not node or node not in STREAM_NODES:
                        continue
                    
                    # Stream Delta (text)
                    text = getattr(msg, "text", None)
                    if text:
                        yield {"type": "delta", "content": text}
                        continue
                    
                    # Stream Delta (str)
                    content = getattr(msg, "content", None)
                    if content:
                        yield {"type": "delta", "content": str(content)}

                # Stream Updates
                elif mode == "updates":
                    data, metadata = payload, {}
                    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
                        data, metadata = payload
                    if not isinstance(data, dict):
                        continue

                    for step, patch in data.items():
                        yield {
                            "type": "updates",
                            "step": step,
                            "content": patch,
                        }

                # Stream Custom
                elif mode == "custom":

                    # Parse Payload
                    data, metadata = payload, {}
                    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
                        data, metadata = payload
                    
                    # Stream Custom Event
                    if isinstance(data, dict) and data.get("type") == "event":
                        yield {"type": "event", "content": data.get("content", "")}
           
            # Completion Event
            yield {"type": "complete"}
        
        # Exception Handling
        except Exception as exc:
            logger.exception("[graph.py] LangGraphProcess processing error")
            raise
        
        # Cleanup
        finally:
            if hasattr(stream, "aclose"):
                await stream.aclose()

    async def run_job(
        self,
        thread_id: str,
        user_id: str,
        user_query: str,
        cancel: Callable[[], Awaitable[bool]] | None = None,
    ):
        """
        JobWorker LangGraphProcess execution wrapper.
        Args:
            thread_id: Thread ID (Session ID)
            user_id: User ID
            user_query: User Query
            cancel (optional): Cancellation callable.
        Yields:
            Streamed Events (Messages, Custom Events)
        """
        async for event in self.main(
            thread_id=thread_id,
            user_id=user_id,
            user_query=user_query,
        ):
            if cancel is not None and await cancel():
                yield {"type": "cancelled", "content": "Job cancelled"}
                yield {"type": "complete"}
                return
            yield event
