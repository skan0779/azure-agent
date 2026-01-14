from __future__ import annotations
import logging
from typing import Any

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

DEFAULT_MESSAGE = (
    "**SYSTEM ERROR OCCURRED.**" 
)

def default_message(
    error: Exception | Any, 
    node: str
) -> str:
    """
    Error message for general errors

    Args:
        error (Exception | Any): The error that occurred
        node (str): The node name where the error occurred
    Returns:
        str: The formatted default error message
    Customs:
        title: "Untitled (System Error)"
    """
    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] SYSTEM ERROR: %s/%s", node, error_type, error)
    
    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (System Error)"})

    # Create Message
    return DEFAULT_MESSAGE
