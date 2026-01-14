from __future__ import annotations
import logging

from langchain_openai.chat_models.base import OpenAIRefusalError

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "**MODEL REFUSAL DETECTED.**"
)

def refusal_message(
    error: OpenAIRefusalError,
    node: str,
) -> str:
    """
    Error message for model refusal

    Args:
        error (OpenAIRefusalError): The refusal error from the model
        node (str): The node name where the refusal occurred
    Returns:
        str: The formatted refusal message
    Customs:
        title: "Untitled (Model Refusal)"
    """
    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] MODEL REFUSAL ERROR: %s/%s", node, error_type, error)

    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (Model Refusal)"})
    
    # Create Message
    return REFUSAL_MESSAGE