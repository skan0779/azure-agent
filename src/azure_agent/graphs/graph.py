import os, yaml, logging, inspect, json
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import ContainerClient
from azure.search.documents import SearchClient

import redis.asyncio as aioredis

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
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

from langchain_tavily import TavilySearch

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

from schemas.state import AgentState

from middlewares.stream import event_stream_before_agent, event_stream_before_model

from tools.azure_ai_search import create_azure_ai_search_tool

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
    """
    def __init__(self) -> None:
        """
        Initalize Application Configuration and dependencies that Performs:
            Load Azure Key Vault Client
            Load Azure Key Vault Secrets (Azure OpenAI, Storage Account, Managed Redis, Postgres, PII, TikToken)
            Create Azure OpenAI Models (Main, Small, Embedding)     
            Create Azure AI Search Client
        """
        # Azure Key Vault
        vault_url = os.getenv("KEY_VAULT_URL")
        if not vault_url:
            logger.error("[graph.py] Failed to load KEY_VAULT_URL from environment variables")
            raise RuntimeError("[graph.py] Failed to load KEY_VAULT_URL from environment variables")
        credential = DefaultAzureCredential()
        secret_client = SecretClient(
            vault_url=vault_url,
            credential=credential,
        )
        
        # Azure OpenAI
        self.AZURE_OPENAI_ENDPOINT = secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value
        self.AZURE_OPENAI_API_KEY = secret_client.get_secret("AZURE-OPENAI-API-KEY").value
        self.AZURE_OPENAI_API_VERSION = secret_client.get_secret("AZURE-OPENAI-API-VERSION").value
        self.AZURE_OPENAI_MAIN_MODEL = secret_client.get_secret("AZURE-OPENAI-MAIN-MODEL").value
        self.AZURE_OPENAI_SMALL_MODEL = secret_client.get_secret("AZURE-OPENAI-SMALL-MODEL").value
        self.AZURE_OPENAI_EMBEDDING_MODEL = secret_client.get_secret("AZURE-OPENAI-EMBEDDING-MODEL").value
        
        # Azure AI Search
        self.AZURE_AI_SEARCH_ENDPOINT = secret_client.get_secret("AZURE-AI-SEARCH-ENDPOINT").value
        self.AZURE_AI_SEARCH_API_KEY = secret_client.get_secret("AZURE-AI-SEARCH-API-KEY").value
        self.AZURE_AI_SEARCH_INDEX_NAME = secret_client.get_secret("AZURE-AI-SEARCH-INDEX-NAME").value
        self.AZURE_AI_SEARCH_SEMANTIC_CONFIG = secret_client.get_secret("AZURE-AI-SEARCH-SEMANTIC-CONFIG").value
        self.AZURE_AI_SEARCH_API_VERSION = secret_client.get_secret("AZURE-AI-SEARCH-API-VERSION").value
        self.AZURE_AI_SEARCH_TOP_K = secret_client.get_secret("AZURE-AI-SEARCH-TOP-K").value
        
        # Blob Storage
        self.BLOB_CONTAINER_NAME = secret_client.get_secret("BLOB-CONTAINER-NAME").value
        self.BLOB_CONNECTION_STRING = secret_client.get_secret("BLOB-CONNECTION-STRING").value
        self.BLOB_CONTAINER_CLIENT = ContainerClient.from_connection_string(
            conn_str=self.BLOB_CONNECTION_STRING,
            container_name=self.BLOB_CONTAINER_NAME,
        )

        # Redis
        self.REDIS_HOST = secret_client.get_secret("REDIS-HOST").value
        self.REDIS_USERNAME = secret_client.get_secret("REDIS-USERNAME").value
        self.REDIS_ACCESS_KEY = secret_client.get_secret("REDIS-ACCESS-KEY").value
        self.REDIS_PORT = secret_client.get_secret("REDIS-PORT").value
        self.REDIS_DB = secret_client.get_secret("REDIS-DB").value

        # Postgres
        self.POSTGRES_CONN_STRING = secret_client.get_secret("POSTGRES-CONN-STRING").value

        # Tavily Search
        os.environ["TAVILY_API_KEY"] = secret_client.get_secret("TAVILY-API-KEY").value

        # PII
        # self.PII_HOST = secret_client.get_secret("PII-HOST").value

        # TikToken
        self.TIKTOKEN_ENCODER = secret_client.get_secret("TIKTOKEN-ENCODER").value
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(__file__).resolve().parent.parent / "encoder")

        # Azure OpenAI Model (MAIN)
        self.main_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_MAIN_MODEL,
            tiktoken_model_name=self.AZURE_OPENAI_MAIN_MODEL,
            model=self.AZURE_OPENAI_MAIN_MODEL,
            stream_usage=True,
            request_timeout=60,

            # GPT 4.x parameters (Optional)
            # temperature=0.2,

            # GPT 5.x parameters (Optional)
            # reasoning_effort="low",
            # verbosity="low",
            # max_completion_tokens=1536,
        )

        # Azure OpenAI Model (SMALL)
        self.small_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_SMALL_MODEL,
            tiktoken_model_name=self.AZURE_OPENAI_SMALL_MODEL,
            model=self.AZURE_OPENAI_SMALL_MODEL,
            streaming=True,
            stream_usage=False,
            request_timeout=60,

            # GPT 4.x parameters (Optional)
            # temperature=0
        )

        # Azure OpenAI Model (Embedding)
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_EMBEDDING_MODEL,
            model=self.AZURE_OPENAI_EMBEDDING_MODEL,
        )

        # Azure AI Search (Client)
        self.azure_ai_search_client = SearchClient(
            endpoint=self.AZURE_AI_SEARCH_ENDPOINT,
            index_name=self.AZURE_AI_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.AZURE_AI_SEARCH_API_KEY),
            api_version=self.AZURE_AI_SEARCH_API_VERSION,
        )

    async def setup(self):
        """
        Initalize Application Runtime Resource that Performs:
            Create Redis Client
            Create Checkpointer (Shallow Redis): ttl 1 day
            Create  Store (Postgres): ttl 30 days
            Build Agent runnable
        """
        # Redis Client
        self.redis_client = aioredis.Redis(
            host=self.REDIS_HOST,
            port=int(self.REDIS_PORT or 10000),
            username=self.REDIS_USERNAME,
            password=self.REDIS_ACCESS_KEY,
            db=int(self.REDIS_DB or 0),
            ssl=True,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Checkpointer
        self.memory = AsyncShallowRedisSaver(
            redis_client=self.redis_client,
            ttl={
                "default_ttl": 60 * 60 * 24,
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
                "default_ttl": 60 * 24 * 30,
                "refresh_on_read": True, 
            },
        )
        
        self._store_cm = store_cm
        self.store = await self._store_cm.__aenter__()

        setup = getattr(self.store, "asetup", None) or getattr(self.store, "setup", None)
        if callable(setup):
            maybe = setup()
            if inspect.isawaitable(maybe):
                await maybe

        # Build Graph
        self.graph = self._build_graph(
            checkpointer=self.memory,
            store=self.store,
        )

    async def close(self) -> None:
        """
        Cleanup Application Runtime Resources that Performs:
            - Stop Store Sweeper
            - Cleanup Store
            - Cleanup Store context manager
            - Cleanup Checkpointer
            - Cleanup Redis Client, Connection Pool
        """
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
            except Exception:
                pass

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
            except Exception:
                pass
            
            # Remove Reference
            self.store = None
        
        # Cleanup Store context manager
        store_cm = getattr(self, "_store_cm", None)
        if store_cm is not None:
            try:
                await store_cm.__aexit__(None, None, None)
            except Exception:
                pass
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
            except Exception:
                pass

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
            except Exception:
                pass
            
            # Cleanup Redis Connection Pool
            try:
                pool = getattr(redis_client, "connection_pool", None)
                if pool is not None:
                    disconnect = getattr(pool, "disconnect", None)
                    if callable(disconnect):
                        maybe = disconnect()
                        if inspect.isawaitable(maybe):
                            await maybe
            except Exception:
                pass

            # Remove Reference
            self.redis_client = None

    def _load_prompt(self, file_name: str) -> str:
        """
        LangGraphProcess Prompt Loader.
        Load a system prompt from Blob Storage or Local File.

        Args:
            file_name: Prompt file name (e.g., "agent_system.yaml").
        Returns:
            system prompt string
        Raises:
            FileNotFoundError: If the local prompt file does not exist.
        """
        # Download from Blob Storage
        try:
            downloader = self.BLOB_CONTAINER_CLIENT.download_blob(file_name)
            raw = downloader.readall().decode("utf-8")

        # Load from Local Repository
        except Exception as exc:
            logger.warning("[graph.py] Failed to load system prompt from blob storage (%s): %s", file_name, exc)
            prompt_path = Path(__file__).resolve().parents[1] / "prompts" / file_name
            try:
                raw = prompt_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.error("[graph.py] Failed to load system prompt from local file: %s", prompt_path)
                raise
        
        data = yaml.safe_load(raw)
        return str(data["system"]).strip()

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
        # Create Azure AI Search Tool
        azure_ai_search_tool = create_azure_ai_search_tool(
            azure_ai_search_client=self.azure_ai_search_client,
            embedding_model=self.embedding_model,
            semantic_config=self.AZURE_AI_SEARCH_SEMANTIC_CONFIG,
            top_k=int(self.AZURE_AI_SEARCH_TOP_K),
        )

        # Create Tavily Search Tool
        tavily_search_tool = TavilySearch(max_results=3, topic="general")
        
        # Create Main Agent
        main_agent = create_agent(
            model=self.main_model,
            tools=[
                azure_ai_search_tool,
                tavily_search_tool,
            ],
            system_prompt=self._load_prompt("main_agent_prompt.yaml"),
            middleware=[
                # Model Middleware
                ModelCallLimitMiddleware(run_limit=2, exit_behavior="end"),
                # ModelRetryMiddleware(max_retries=1),
                # ModelFallbackMiddleware(self.small_model),
                
                # Tool Middleware
                ToolCallLimitMiddleware(run_limit=1, exit_behavior="continue"),
                # ToolRetryMiddleware(max_retries=1),
                
                # Message Middleware
                SummarizationMiddleware(
                    model=self.small_model,
                    trigger=[("tokens", 20000)],
                    keep=("messages", 20),
                    token_counter=self.small_model.get_num_tokens_from_messages,
                ),

                # # PII Middleware
                PIIMiddleware("email", strategy="mask"),
                PIIMiddleware("credit_card", strategy="mask"),
                PIIMiddleware("ip", strategy="redact"),
                PIIMiddleware("mac_address", strategy="redact"),
                PIIMiddleware("url", strategy="redact"),

                # # Custom Middleware
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
            configurable={"thread_id": thread_id},
        )

        # Stream Processing
        stream = self.graph.astream(
            inputs, 
            config,
            subgraphs=True,
            stream_mode=[
                "messages", 
                "updates", 
                "custom",
                # "debug" # (optional)
            ],
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

                    # Stream Custom Title
                    if isinstance(data, dict) and data.get("type") == "title":
                        yield {"type": "title", "content": data.get("content", "")}
            
                # Stream Debug (optional)
                # elif mode == "debug":
                #     e = payload
                #     if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], dict):
                #         e = payload[0]
                #     if not isinstance(e, dict):
                #         logger.info("[debug] raw=%s", e)
                #         continue

                #     et = e.get("type")
                #     step = e.get("step")
                #     pl = e.get("payload") or {}
                #     name = pl.get("name")

                #     # task: 노드 실행 시작
                #     if et == "task":
                #         logger.info("[debug] step=%s task name=%s triggers=%s", step, name, pl.get("triggers"))

                #     # task_result: 노드 실행 결과
                #     elif et == "task_result":
                #         err = pl.get("error")
                #         logger.info("[debug] step=%s result name=%s error=%s", step, name, bool(err))
                #     else:
                #         logger.info("[debug] step=%s type=%s payload_keys=%s", step, et, list(pl.keys()) if isinstance(pl, dict) else type(pl).__name__)

            # Completion Event
            yield {"type": "complete"}
        
        # Exception Handling
        except Exception as exc:
            logger.error("[graph.py] LangGraphProcess processing error : %s", exc)
        
        # Cleanup (per requests)
        finally:
            if hasattr(stream, "aclose"):
                await stream.aclose()
