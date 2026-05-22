import asyncio, os, yaml, logging, inspect, base64
from pathlib import Path
from collections.abc import Awaitable, Callable

from azure.storage.blob.aio import ContainerClient
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from redis.asyncio import Redis

from fastapi.encoders import jsonable_encoder

from langchain_core.messages import HumanMessage, SystemMessage, message_to_dict
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_azure_dynamic_sessions.backends import SessionsBashBackend
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    PIIMiddleware,
)
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.store.postgres import AsyncPostgresStore, PoolConfig
from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data

from azure_agent.infra.key_vault import create_async_secret_client
from azure_agent.config import AppSecrets, load_app_secrets
from azure_agent.files import AgentFileRepository
from azure_agent.graphs.schema import AgentContext
from azure_agent.tools.azure_ai_search import create_azure_ai_search_tool
from azure_agent.tools.sessions_python_repl import create_python_repl_tool
from azure_agent.middlewares.azure_ai_content_safety import (
    azure_content_moderation_middleware,
    azure_prompt_shield_middleware,
)
from azure_agent.middlewares.azure_dynamic_session import (
    SessionsFileSyncMiddleware,
)

logger = logging.getLogger(__name__)


class LangGraphProcess:
    """
    LangGraphProcess application configuration and runtime resource manager:
    - Load Azure Key Vault Secrets
    - Create Azure OpenAI models (main, small, embedding)
    - Create Azure AI Search vectorstore and retriever-backed tool
    - Create Redis Client
    - Create Checkpointer (Redis)
    - Create Store (Postgres)
    - Build Agent runnable
    """
    def __init__(self, vault_url: str | None = None) -> None:
        """
        Initialize lightweight application state.
        """
        self.vault_url = vault_url
        self.secret_client: SecretClient | None = None
        self.secret_credential: DefaultAzureCredential | None = None
        self.secrets: AppSecrets | None = None
        self.prompt_cache: dict[str, str] = {}

        # Runtime clients / models
        self.BLOB_CONTAINER_CLIENT: ContainerClient | None = None
        self.FILES_BLOB_CONTAINER_CLIENT: ContainerClient | None = None
        self.agent_file_repository: AgentFileRepository | None = None
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
        Initialize application runtime resources.

        Responsibilities:
            Load Azure Key Vault Secrets
            Create Azure OpenAI models (main, small, embedding)
            Create Azure AI Search vectorstore
            Create Blob Client
            Create Redis Client
            Create Checkpointer (Shallow Redis): ttl 1 day
            Create Store (Postgres): ttl 30 days
            Load Prompt Cache
            Build Agent runnable
        """
        await self._load_secrets()
        self._load_instance()

        # Open file repository connection pool
        if self.agent_file_repository is not None:
            await self.agent_file_repository.open()

        # Redis Client
        if self.secrets is None:
            raise RuntimeError("Secrets are not loaded")
        secrets = self.secrets
        self.redis_client = Redis(
            host=secrets.REDIS_HOST,
            port=int(secrets.REDIS_PORT),
            username=secrets.REDIS_USERNAME,
            password=secrets.REDIS_ACCESS_KEY,
            db=int(secrets.REDIS_DB),
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
            conn_string=secrets.POSTGRES_CONN_STRING,
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
                "dims": int(secrets.AZURE_OPENAI_EMBEDDING_DIMS),
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

        # Seed skills
        await self._seed_skills(
            agent_name="main_agent",
            source_dir=Path(__file__).resolve().parents[1] / "skills" / "langchain-skills"
        )

        # Load Prompt
        await self._load_prompts([
            "main_agent.yaml",
            "sandbox_agent.yaml",
        ])

        # Build Graph
        self.graph = self._build_graph(
            checkpointer=self.memory,
            store=self.store,
        )

    async def _load_secrets(self) -> None:
        self.secret_client, self.secret_credential = create_async_secret_client(
            self.vault_url
        )

        assert self.secret_client is not None
        self.secrets = await load_app_secrets(self.secret_client)

    def _load_instance(self) -> None:
        # Load Secrets
        if self.secrets is None:
            raise RuntimeError("Secrets are not loaded")
        secrets = self.secrets

        # Initialize Blob Container Client
        self.BLOB_CONTAINER_CLIENT = ContainerClient.from_connection_string(
            conn_str=secrets.BLOB_CONNECTION_STRING,
            container_name="prompts",
        )
        self.FILES_BLOB_CONTAINER_CLIENT = ContainerClient.from_connection_string(
            conn_str=secrets.BLOB_CONNECTION_STRING,
            container_name="files",
        )
        self.agent_file_repository = AgentFileRepository(
            conn_string=secrets.POSTGRES_WEB_CONN_STRING,
        )


        # Initialize Azure OpenAI Models
        self.main_model = ChatOpenAI(
            model=secrets.AZURE_OPENAI_MAIN_MODEL,
            base_url=f"{secrets.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
            api_key=secrets.AZURE_OPENAI_API_KEY,
            use_responses_api=True,
            stream_usage=True,
            timeout=int(secrets.AZURE_OPENAI_MAIN_MODEL_TIMEOUT),
        )
        self.small_model = AzureChatOpenAI(
            azure_endpoint=secrets.AZURE_OPENAI_ENDPOINT,
            api_key=secrets.AZURE_OPENAI_API_KEY,
            api_version=secrets.AZURE_OPENAI_API_VERSION,
            azure_deployment=secrets.AZURE_OPENAI_SMALL_MODEL,
            tiktoken_model_name=secrets.AZURE_OPENAI_SMALL_MODEL,
            model=secrets.AZURE_OPENAI_SMALL_MODEL,
            streaming=True,
            stream_usage=False,
            request_timeout=int(secrets.AZURE_OPENAI_SMALL_MODEL_TIMEOUT),
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=secrets.AZURE_OPENAI_ENDPOINT,
            api_key=secrets.AZURE_OPENAI_API_KEY,
            api_version=secrets.AZURE_OPENAI_API_VERSION,
            azure_deployment=secrets.AZURE_OPENAI_EMBEDDING_MODEL,
            model=secrets.AZURE_OPENAI_EMBEDDING_MODEL,
        )
        
        # Initialize Azure AI Search Vectorstore
        self.azure_search = AzureSearch(
            azure_search_endpoint=secrets.AZURE_AI_SEARCH_ENDPOINT,
            azure_search_key=secrets.AZURE_AI_SEARCH_API_KEY,
            index_name=secrets.AZURE_AI_SEARCH_INDEX_NAME,
            embedding_function=self.embedding_model,
            search_type="semantic_hybrid",
            semantic_configuration_name=secrets.AZURE_AI_SEARCH_SEMANTIC_CONFIG,
            vector_search_dimensions=int(secrets.AZURE_OPENAI_EMBEDDING_DIMS),
            additional_search_client_options={
                "api_version": secrets.AZURE_AI_SEARCH_API_VERSION,
            },
        )

    async def _load_prompts(self, file_names: list[str]) -> None:
        for file_name in file_names:
            # Check Cache
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

    async def _seed_memories(self, user_id: str) -> None:
        if self.store is None:
            raise RuntimeError("Store is not initialized")

        key = "/memories/AGENTS.md"
        namespace = ("memories", user_id)

        existing = await self.store.aget(namespace, key)
        if existing is not None:
            return

        memory_path = Path(__file__).resolve().parents[1] / "memories" / "AGENTS.md"
        initial_memory = await asyncio.to_thread(
            memory_path.read_text,
            encoding="utf-8",
        )

        await self.store.aput(
            namespace,
            key,
            create_file_data(initial_memory),
            index=False,
        )

    async def _seed_skills(self, agent_name: str, source_dir: Path) -> None:
        if self.store is None:
            raise RuntimeError("Store is not initialized")

        namespace = ("skills", agent_name)

        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(source_dir).as_posix()
            key = f"/skills/{relative}"

            raw = await asyncio.to_thread(path.read_bytes)
            try:
                content = raw.decode("utf-8")
                file_data = create_file_data(content, encoding="utf-8")
            except UnicodeDecodeError:
                content = base64.standard_b64encode(raw).decode("ascii")
                file_data = create_file_data(content, encoding="base64")

            await self.store.aput(
                namespace,
                key,
                file_data,
                index=False,
            )

    async def close(self) -> None:
        """
        Cleanup application runtime resources.

        Responsibilities:
            - Stop Store Sweeper
            - Cleanup Store
            - Cleanup Store context manager
            - Cleanup Checkpointer
            - Cleanup Redis Client, Connection Pool
            - Cleanup Azure AI Search vectorstore clients
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

        # Cleanup Files Blob Client
        files_blob_container_client = getattr(self, "FILES_BLOB_CONTAINER_CLIENT", None)
        if files_blob_container_client is not None:
            try:
                close = getattr(files_blob_container_client, "close", None)
                if callable(close):
                    maybe = close()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as exc:
                logger.warning("[graph.py] Failed to close files blob container client: %s", exc)
            self.FILES_BLOB_CONTAINER_CLIENT = None

        # Drop Repository Reference
        agent_file_repository = getattr(self, "agent_file_repository", None)
        if agent_file_repository is not None:
            try:
                await agent_file_repository.close()
            except Exception as exc:
                logger.warning(
                    "[graph.py] Failed to close agent file repository pool: %s",
                    exc,
                )
            self.agent_file_repository = None

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
        Build and return the deep agent runnable.
        Args:
            checkpointer: Checkpoint saver instance
            store: Data store instance

        Returns:
            Runnable: Deep agent runnable
        """
        # Load Secret
        if self.secrets is None:
            raise RuntimeError("Secrets are not loaded")
        secrets = self.secrets

        # Create Azure AI Search retriever tool
        azure_ai_search = create_azure_ai_search_tool(
            azure_ai_search=self.azure_search,
            top_k=int(secrets.AZURE_AI_SEARCH_TOP_K),
        )
        
        # Create Azure Dynamic Sessions Python REPL tool
        python_repl_tool = create_python_repl_tool(
            pool_management_endpoint=secrets.AZURE_DYNAMIC_SESSIONS_PYTHON_POOL_ENDPOINT
        )

        # Create Web Search Tool (Grounding with bing search)
        web_search = {
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "KR",
            },
        }

        # Create Sandbox Agent
        sandbox_agent = create_deep_agent(
            model=self.main_model,
            tools=[
                python_repl_tool
            ],
            system_prompt=self.prompt_cache["sandbox_agent.yaml"],
            middleware=[
                SessionsFileSyncMiddleware(
                    pool_management_endpoint=secrets.AZURE_DYNAMIC_SESSIONS_BASH_POOL_ENDPOINT,
                    blob_container_client=self.FILES_BLOB_CONTAINER_CLIENT,
                    file_repository=self.agent_file_repository,
                )
            ],
            context_schema=AgentContext,
            checkpointer=checkpointer,
            store=store,
            backend=lambda rt: CompositeBackend(
                default=SessionsBashBackend(
                    pool_management_endpoint=secrets.AZURE_DYNAMIC_SESSIONS_BASH_POOL_ENDPOINT,
                    session_id=f"sandbox-{rt.context.user_id}-{rt.context.thread_id}",
                ),
                routes={}
             ),
            name="sandbox_agent",
        )

        # Create Main Agent
        main_agent = create_deep_agent(
            model=self.main_model,
            tools=[
                azure_ai_search,
                web_search,
            ],
            system_prompt=self.prompt_cache["main_agent.yaml"],
            middleware=[
                # Model Middleware
                ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),

                # Tool Middleware
                ToolCallLimitMiddleware(run_limit=5, exit_behavior="continue"),

                # PII Middleware
                PIIMiddleware("email", strategy="mask"),
                PIIMiddleware("credit_card", strategy="mask"),
                
                # Content Safety Middleware
                azure_content_moderation_middleware(
                    endpoint=secrets.AZURE_AI_CONTENT_SAFETY_ENDPOINT,
                    credential=secrets.AZURE_AI_CONTENT_SAFETY_API_KEY,
                ),
                azure_prompt_shield_middleware(
                    endpoint=secrets.AZURE_AI_CONTENT_SAFETY_ENDPOINT,
                    credential=secrets.AZURE_AI_CONTENT_SAFETY_API_KEY,
                )
            ],
            subagents=[
                CompiledSubAgent(
                    name="sandbox_agent",
                    description=(
                        "Use for any task involving uploaded files, attachments, /mnt/data files, "
                        "file conversion, file extraction, file editing, generated downloadable files, "
                        "code execution, command-line work, computation, or data analysis. "
                        "Uploaded user files are available only in this sandbox workspace."
                    ),
                    runnable=sandbox_agent,
                )
            ],
            skills=[
                "/skills/",
            ],
            memory=[
                "/memories/AGENTS.md",
            ],
            context_schema=AgentContext,
            checkpointer=checkpointer,
            store=store,
            backend=lambda rt: CompositeBackend(
                default=StateBackend(rt),
                routes={
                    "/memories/": StoreBackend(rt, namespace=lambda rt: ("memories", rt.context.user_id)),
                    "/skills/": StoreBackend(rt, namespace=lambda rt: ("skills", "main_agent")),
                },
            ),
            name="main_agent",
        )

        return main_agent

    async def main(
        self,
        thread_id: str,
        job_id: str,
        user_id: str,
        user_query: str,
    ):
        """
        LangGraphProcess Main Function

        Args:
            thread_id: Thread ID (Session ID)
            job_id: Job ID
            user_id: User ID
            user_query: User Query
        Yields:
            Streamed LangGraph chunks and lifecycle events in `type/ns/data` format
        """
        
        # Logging
        logger.info("[graph.py] LangGraphProcess Request : thread_id=%s, job_id=%s, user_id=%s, user_query=%s", thread_id, job_id, user_id, user_query)

        # Input Values
        inputs = {
            "messages": [
                HumanMessage(content=user_query),
            ],
        }

        # Deep agents expect per-run identity data through runtime context.
        context = AgentContext(
            thread_id=thread_id,
            job_id=job_id, 
            user_id=user_id,
        )

        # Runnable Config
        config = RunnableConfig(
            recursion_limit=30,
            configurable={
                "thread_id": thread_id,
                "job_id": job_id,
                "user_id": user_id,
            },
        )

        await self._seed_memories(user_id)

        # Stream Processing
        stream = self.graph.astream(
            input=inputs,
            config=config,
            context=context,
            subgraphs=True,
            stream_mode=[
                "messages",
                "updates",
                "custom",
                # "tasks" # optional
            ],
            version="v2",
        )
        try:
            async for chunk in stream:
                # Stream Messages
                if chunk["type"] == "messages":
                    msg, metadata = chunk["data"]
                    yield {
                        "type": "messages",
                        "ns": list(chunk["ns"]),
                        "data": [
                            message_to_dict(msg),
                            jsonable_encoder(metadata),
                        ],
                    }

                # Stream Updates
                elif chunk["type"] == "updates":
                    yield {
                        "type": "updates",
                        "ns": list(chunk["ns"]),
                        "data": jsonable_encoder(chunk["data"]),
                    }

                # Stream Custom
                elif chunk["type"] == "custom":
                    yield {
                        "type": "custom",
                        "ns": list(chunk["ns"]),
                        "data": jsonable_encoder(chunk["data"]),
                    }

                # Stream Tasks
                # elif chunk["type"] == "tasks":
                #     yield {
                #         "type": "tasks",
                #         "ns": list(chunk["ns"]),
                #         "data": jsonable_encoder(chunk["data"]),
                #     }
           
            # Completion Event
            yield {
                "type": "complete",
                "ns": [],
                "data": None
            }
        
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
        job_id: str,
        user_id: str,
        user_query: str,
        cancel: Callable[[], Awaitable[bool]] | None = None,
    ):
        """
        JobWorker LangGraphProcess execution wrapper.
        Args:
            thread_id: Thread ID (Session ID)
            job_id: Job ID
            user_id: User ID
            user_query: User Query
            cancel (optional): Cancellation callable.
        Yields:
            Streamed LangGraph chunks and lifecycle events in `type/ns/data` format
        """
        async for event in self.main(
            thread_id=thread_id,
            job_id=job_id,
            user_id=user_id,
            user_query=user_query,
        ):
            if cancel is not None and await cancel():
                yield {
                    "type": "cancelled",
                    "ns": [],
                    "data": {"message": "Job cancelled"},
                }
                yield {
                    "type": "complete", 
                    "ns": [], 
                    "data": None
                }
                return
            yield event
