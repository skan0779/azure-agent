from __future__ import annotations
from typing import Any
import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

PARSER_ERROR_MESSAGE = (
    "**MODEL PARSING FAILED.**"
)

def parser_error_message(
    error: Any, 
    node: str
) -> str:
    """
    Error message for model parsing errors

    Args:
        error (Any): The exception or error encountered
        node (str): The node name where the error occurred
    Returns:
        str: The formatted model parsing error message
    Customs:
        title: "Untitled (Parsing Failed)"
    """
    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] MODEL PARSING FAILED: %s/%s", node, error_type, error)
    
    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (Parsing Failed)"})
    
    # Create Message
    return PARSER_ERROR_MESSAGE