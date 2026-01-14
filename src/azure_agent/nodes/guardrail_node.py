import httpx
import tiktoken
import logging

from langchain_core.messages import HumanMessage, AIMessage

from langgraph.config import get_stream_writer

from messages import token_limit_message, pii_message, pii_error_message

from schemas.state import AgentState

logger = logging.getLogger(__name__)

def create_guardrail_node(
    *, 
    TOKEN_LIMIT: int = 30000,
    ENCODER: str = "o200k_base",
    PII_HOST: str = "http://localhost:8000",
    PII_CHUNK: int = 10000,
):
    """
    Create a node that performs:
        - Token Limit Check
        - Personal Information Identification Check
    
    Messages:
        token_limit_message
        pii_message
        pii_error_message (httpx.HTTPStatusError, httpx.RequestError, Exception)
    Customs:
        event: "counting tokens..."
        event: "checking PII (1/n)..."
    Args:
        TOKEN_LIMIT: Maximum allowed token count for user input (default: 30000)
        ENCODER: Token encoder name (default: "o200k_base")
        PII_HOST: Host URL for the PII checking service (default: "http://localhost:8000")
        PII_CHUNK: Token chunk size for PII checking (default: 10000)
    Raises:
        httpx.HTTPStatusError: If the PII service returns an error status.
        httpx.RequestError: If there is an error while making the HTTP request.
        Exception: For any other exceptions that may occur.
    """
    NODE = "guardrail_node"

    def _mask_message(state: AgentState, masked_text: str) -> None:
        """
        Helper Function for making the last human message in the state with the provided masked text for PII compliance.
        
        Args:
            state: The current agent state containing messages.
            masked_text: The text to replace the last human message content with.
        """
        messages = state.get("messages") or []
        if not messages:
            return

        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            last_msg.content = masked_text

    async def guardrail_node(state: AgentState, **_):

        # User Query
        user_query = (state.get("user_query") or "").strip()
        
        # Event Streaming
        writer = get_stream_writer()
        writer({"type": "event", "content": "counting tokens..."})

        # Token Limit Handling
        encoding = tiktoken.get_encoding(ENCODER)
        tokens = encoding.encode(user_query)
        total_tokens = len(tokens)
        if total_tokens > TOKEN_LIMIT:
            msg = token_limit_message(total_tokens, TOKEN_LIMIT, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg)]}
        
        # Personal Information Identification Handling
        try:
            # Single-chunk Processing
            if total_tokens <= PII_CHUNK:
                
                # Event Streaming
                writer({"type": "event", "content": "checking PII..."})

                # HTTP Request
                async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=2.0)) as client:
                    resp = await client.post(PII_HOST + "/api/pii", json={"text": user_query})
                
                resp.raise_for_status()
                data = resp.json()
                blocked: bool = bool(data.get("blocked", False))
                label_list = data.get("label_list", []) or []
                masked_user_query: str = data.get("masked_text", user_query)
                
                # PII Detected
                if blocked:
                    
                    # Update State Message
                    _mask_message(state, masked_user_query)

                    # Create Message
                    msg = pii_message(labels=label_list, node=NODE)
                    
                    return {"guardrail": True, "user_query": masked_user_query, "messages": [AIMessage(content=msg)]}
                else:
                    return {"guardrail": False}

            # Multi-chunk Processing
            else:
                masked_chunks = []
                async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=2.0)) as client:
                    for i in range(0, total_tokens, PII_CHUNK):
                        
                        # Event Streaming
                        writer({"type": "event", "content": f"checking PII ({i//PII_CHUNK + 1}/{(total_tokens-1)//PII_CHUNK + 1})..."})

                        # Chunk Processing
                        chunk_tokens = tokens[i:i + PII_CHUNK]
                        chunk_text = encoding.decode(chunk_tokens)

                        # HTTP Request
                        resp = await client.post(PII_HOST + "/api/pii", json={"text": chunk_text})
                        resp.raise_for_status()
                        data = resp.json()
                        chunk_blocked = bool(data.get("blocked", False))
                        chunk_labels = data.get("label_list", []) or []
                        chunk_masked = data.get("masked_text", chunk_text)
                        masked_chunks.append(chunk_masked)

                        # PII Detected in Chunk
                        if chunk_blocked:

                            # Combine Masked Chunks
                            masked_user_query = "".join(masked_chunks) + "..."

                            # Update State Message
                            _mask_message(state, masked_user_query)

                            # Create Message
                            msg = pii_message(labels=chunk_labels, node=NODE)

                            return {"guardrail": True, "user_query": masked_user_query, "messages": [AIMessage(content=msg)]}
                        
                return {"guardrail": False}

        # HTTP Status Error Handling
        except httpx.HTTPStatusError as e:
            msg = pii_error_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg)]}

        # HTTP Request Error Handling
        except httpx.RequestError as e:
            msg = pii_error_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg)]}
        
        # Other Exception Handling
        except Exception as e:
            msg = pii_error_message(e, NODE)          
            return {"guardrail": True, "messages": [AIMessage(content=msg)]}

    return guardrail_node
