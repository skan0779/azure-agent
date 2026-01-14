from __future__ import annotations
from typing import Any
import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

PII_ERROR_MESSAGE = (
    "**PII CHECKING FAILED.**"
)

def pii_error_message(
    error: Any, 
    node: str
) -> str:
    """
    Error message for PII check failures

    Args:
        error (Any): The exception or error encountered
        node (str): The node name where the error occurred
    Returns:
        str: The formatted PII error message
    Customs:
        title: "Untitled (PII)"
    """
    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] PII CHECKING FAILED: %s/%s", node, error_type, error)
    
    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (PII)"})
    
    # Create Message
    return PII_ERROR_MESSAGE
