from __future__ import annotations
import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

MAX_TOKEN_MESSAGE = (
    "**TOKEN LIMIT EXCEEDED.({total_tokens}/{token_limit})**"
)

def token_limit_message(
    token_limit: int, 
    total_tokens: int, 
    node: str
) -> str:
    """
    Error message for exceeding token limit

    Args:
        token_limit (int): The maximum allowed token limit
        total_tokens (int): The total tokens counted
        node (str): The node name where the limit was exceeded
    Returns:
        str: The formatted token limit exceeded message
    Customs:
        title: "Untitled (Token Limit)"
    """
    # Logging
    logger.warning("[%s] TOKEN LIMIT EXCEEDED: %d/%d", node, total_tokens, token_limit)
    
    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (Token Limit)"})

    # Create Message
    return MAX_TOKEN_MESSAGE.format(
        total_tokens=total_tokens,
        token_limit=token_limit
    )
