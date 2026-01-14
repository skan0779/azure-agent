import os
import yaml
import logging
import inspect
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import ContainerClient
from azure.search.documents import SearchClient

import redis.asyncio as aioredis

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.graph import StateGraph, START, END

from schemas.state import AgentState

from tools.azure_ai_search import create_azure_ai_search_tool

from edges.router_edge import router_conditional_edge
from edges.guardrail_edge import guardrail_conditional_edge

from nodes.router_node import create_router_node
from nodes.guardrail_node import create_guardrail_node

logger = logging.getLogger(__name__)


class LangGraphProcess:

    def __init__(self) -> None:
        """
        Initalize Application Configuration and dependencies that Performs:
            Load Azure Key Vault Client
            Load Azure Key Vault Secrets (Azure OpenAI, Storage Account, Managed Redis)
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
        self.BLOB_NAME = secret_client.get_secret("BLOB-PROMPTS").value
        self.BLOB_CONNECTION_STRING = secret_client.get_secret("BLOB-CONNECTION-STRING").value
        self.BLOB_CONTAINER_CLIENT = ContainerClient.from_connection_string(
            conn_str=self.BLOB_CONNECTION_STRING,
            container_name=self.BLOB_NAME,
        )

        # Redis
        self.REDIS_HOST = secret_client.get_secret("REDIS-HOST").value
        self.REDIS_ACCESS_KEY = secret_client.get_secret("REDIS-ACCESS-KEY").value
        self.REDIS_PORT = secret_client.get_secret("REDIS-PORT").value
        self.REDIS_DB = secret_client.get_secret("REDIS-DB").value

        # PII
        self.PII_HOST = secret_client.get_secret("PII-HOST").value

        # TikToken
        self.TIKTOKEN_ENCODER = secret_client.get_secret("TIKTOKEN-ENCODER").value
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(__file__).resolve().parent.parent / "encoder")

        # Azure OpenAI Model (MAIN)
        self.main_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_MAIN_MODEL,
            model=self.AZURE_OPENAI_MAIN_MODEL,
            stream_usage=True,
        )

        # Azure OpenAI Model (SMALL)
        self.small_model = AzureChatOpenAI(
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_deployment=self.AZURE_OPENAI_SMALL_MODEL,
            model=self.AZURE_OPENAI_SMALL_MODEL,
            stream_usage=False,
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
            Create Checkpointer
            Compile StateGraph
        """
        # Redis Client
        self.redis_client = aioredis.Redis(
            host=self.REDIS_HOST,
            port=int(self.REDIS_PORT or 10000),
            password=self.REDIS_ACCESS_KEY,
            db=int(self.REDIS_DB or 0),
            ssl=True,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Checkpointer
        self.memory = AsyncShallowRedisSaver(redis_client=self.redis_client)
        await getattr(self.memory, "asetup", lambda: None)()

        # Build Graph
        self.graph = self._build_graph(checkpointer=self.memory)

    async def close(self) -> None:
        """
        Cleanup Application Runtime Resources that Performs:
            Cleanup Checkpinter
            Cleanup Redis (Client, Connection Pool)
        """
        # Cleanup Checkpointer
        memory = getattr(self, "memory", None)
        if memory is not None:
            aclose = getattr(memory, "aclose", None)
            close = getattr(memory, "close", None)
            if callable(aclose):
                await aclose()
            elif callable(close):
                maybe = close()
                if inspect.isawaitable(maybe):
                    await maybe

        # Cleanup Redis Client
        redis_client = getattr(self, "redis_client", None)
        if redis_client is not None:

            # Cleanup Redis Client
            aclose = getattr(redis_client, "aclose", None)
            close = getattr(redis_client, "close", None)
            if callable(aclose):
                await aclose()
            elif callable(close):
                maybe = close()
                if inspect.isawaitable(maybe):
                    await maybe
            
            # Cleanup Redis Connection Pool
            pool = getattr(redis_client, "connection_pool", None)
            if pool is not None:
                disconnect = getattr(pool, "disconnect", None)
                if callable(disconnect):
                    maybe = disconnect()
                    if inspect.isawaitable(maybe):
                        await maybe

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

    def _build_graph(self, checkpointer=None):
        """
        LangGraphProcess StateGraph Compiler that builds:
            State:
                AgentState: TypedDict Schema
            Tools:
                azure_ai_search_tool: RAG Tool
            Nodes:
                Guardrail: Token limit & PII
                Router: Routing to agents
                Main Agent: Main Conversational Agent
                Document Agent: Document Generation Agent
            Edges:
                Guardrail Conditional Edge
                Router Conditional Edge
        
        Args:
            checkpointer: checkpoint saver Instance
        Returns:
            StateGraph: LangGraphProcess StateGraph Instance
        """
        # Build Graph
        builder = StateGraph(AgentState)

        # Create Azure AI Search Tool
        azure_ai_search_tool = create_azure_ai_search_tool(
            azure_ai_search_client=self.azure_ai_search_client,
            embedding_model=self.embedding_model,
            semantic_config=self.AZURE_AI_SEARCH_SEMANTIC_CONFIG,
            top_k=int(self.AZURE_AI_SEARCH_TOP_K),
        )

        # Create Guardrail Node
        guardrail_node = create_guardrail_node(
            ENCODER=self.TIKTOKEN_ENCODER,
            TOKEN_LIMIT=30000,
            PII_HOST=self.PII_HOST,
            PII_CHUNK=10000,
        )
        builder.add_node("guardrail", guardrail_node)

        # Create Router Node
        router_node = create_router_node(
            model=self.small_model,
            system_prompt=self._load_prompt("router_prompt.yaml"),
        )
        builder.add_node("router", router_node)
        
        # Create Main Agent
        main_agent = create_agent(
            model=self.main_model,
            tools=[azure_ai_search_tool],
            system_prompt=self._load_prompt("main_agent_prompt.yaml"),
            middleware=[],
            state_schema=AgentState,
            name="main_agent",
        )
        builder.add_node("main_agent", main_agent)

        # Create Document Agent
        document_agent = create_agent(
            model=self.main_model,
            tools=[],
            system_prompt=self._load_prompt("document_agent_prompt.yaml"),
            middleware=[],
            state_schema=AgentState,
            name="document_agent",
        )
        builder.add_node("document_agent", document_agent)

        # Define Edges and Conditonal Edges
        builder.add_edge(START, "guardrail")
        builder.add_conditional_edges(
            "guardrail",
            guardrail_conditional_edge,
            {
                "router":"router",
                END:END,
            },
        )
        builder.add_conditional_edges(
            "router",
            router_conditional_edge,
            {
                "main_agent": "main_agent",
                "document_agent": "document_agent",
                END:END,
            },
        )
        builder.add_edge("main_agent", END)
        builder.add_edge("document_agent", END)

        return builder.compile(checkpointer=checkpointer)

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
            recursion_limit=20,
            configurable={"thread_id": thread_id},
        )

        # Stream Processing
        stream = self.graph.astream(inputs, config, subgraphs=True, stream_mode=["messages", "custom"])
        try:
            async for event in stream:

                # Parse Event
                if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
                    mode, payload = event
                else:
                    mode, payload = "messages", event

                # Stream Messages
                if mode == "messages":
                    
                    # Stream Specific Nodes
                    STREAM_BLOCKED_NODES = ["router"]

                    # Parse Payload
                    msg, metadata = payload if isinstance(payload, tuple) else (payload, None)
                    
                    # Stream Specific Messages
                    additional_kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
                    node = (metadata or {}).get("langgraph_node")
                    stream = bool(additional_kwargs.get("stream") or False)
                    if node in STREAM_BLOCKED_NODES and not stream:
                        continue
                    else:
                        content = getattr(msg, "content", None)
                        if content:
                            yield {"type": "delta", "content": str(content)}

                # Stream Custom Events
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
            
            # Completion Event
            yield {"type": "complete"}
        
        # Exception Handling
        except Exception as exc:
            logger.error("[graph.py] LangGraphProcess processing error : %s", exc)
        
        # Cleanup (per requests)
        finally:
            if hasattr(stream, "aclose"):
                await stream.aclose()
