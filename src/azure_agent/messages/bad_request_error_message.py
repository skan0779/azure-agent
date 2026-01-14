from __future__ import annotations
import ast
import logging
from typing import Any, Dict

from openai import BadRequestError

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

CONTENT_SAFETY_MESSAGE = (
    "**CONTENT SAFETY({categories}) DETECTED.**"
)

CONTEXT_LENGTH_MESSAGE = (
    "**CONTEXT LENGTH EXCEEDED.**"
)

INVALID_REQUEST_MESSAGE = (
    "**INVALID REQUEST DETECTED.**"
)

BAD_REQUEST_MESSAGE = (
    "**BAD REQUEST ERROR DETECTED.**"
)

def _extract_payload(error: BadRequestError) -> Dict[str, Any]:
    """
    Helper Function to extract payload from BadRequestError
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body

    text = str(error)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        literal = text[start : end + 1]
        try:
            parsed = ast.literal_eval(literal)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

    return {}

def bad_request_error_message(
    error: BadRequestError,
    node: str,
    payload: Dict[str, Any] | None = None,
) -> str:
    """
    Warning/Error messages for BadRequestError

    Args:
        error (BadRequestError): The BadRequestError from the model
        node (str): The node name where the bad request error was detected
        payload (Dict[str, Any] | None): Optional payload extracted from the error
    Returns:
        str: The formatted error message
    Customs:
        title: "Untitled (Content Safety)"
        title: "Untitled (Context Length Exceeded)"
        title: "Untitled (Invalid Request)"
        title: "Untitled (Bad Request Error)"
    """
    # Check Error Code
    payload = payload if payload is not None else _extract_payload(error)
    error_obj = payload.get("error") if isinstance(payload, dict) else {}
    if not isinstance(error_obj, dict):
        error_obj = {}
    code = error_obj.get("code")
    code = code.lower() if isinstance(code, str) else ""

    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] BAD REQUEST ERROR DETECTED: %s/%s", node, error_type, error)

    # Content Safety Detected
    if code == "content_filter":

        # Title Generation
        writer = get_stream_writer()
        writer({"type": "title", "content": "Untitled (Content Safety)"})

        # Create Message
        inner_error = error_obj.get("innererror") or {}
        if not isinstance(inner_error, dict):
            inner_error = {}
        filter_result = inner_error.get("content_filter_result") or {}
        if not isinstance(filter_result, dict):
            filter_result = {}
        filtered_categories = [
            category
            for category, detail in filter_result.items()
            if isinstance(detail, dict) and detail.get("filtered")
        ]
        categories_text = ", ".join(filtered_categories) if filtered_categories else "Blocked Content"
        
        return CONTENT_SAFETY_MESSAGE.format(categories=categories_text)

    # Context Length Exceeded Detected
    elif code == "context_length_exceeded":

        # Title Generation
        writer = get_stream_writer()
        writer({"type": "title", "content": "Untitled (Context Length Exceeded)"})

        # Create Message
        return CONTEXT_LENGTH_MESSAGE
    
    # Invalid Request Detected
    elif code in {"unsupported_parameter", "invalid_value", "invalidpayload", "invalid_request_error"}:
        
        # Title Generation
        writer = get_stream_writer()
        writer({"type": "title", "content": "Untitled (Invalid Request)"})

        # Create Message
        return INVALID_REQUEST_MESSAGE
    
    # Default Bad Request Message
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (Bad Request Error)"})

    # Create Message
    return BAD_REQUEST_MESSAGE