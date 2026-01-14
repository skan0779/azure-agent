from __future__ import annotations
from typing import Any, Dict
import logging

from openai import RateLimitError

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

RATE_LIMIT_MESSAGE = (
    "**TOKEN RATE LIMIT EXCEEDED.(Retry after: {retry_after}s, {remaining_tokens}/{limit_tokens})**"
)

def _get_retry_after(headers: dict) -> str:
    """
    Helper function to extract retry-after seconds from header
    """
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    retry_after_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")

    if retry_after is not None:
        return str(retry_after)

    if retry_after_ms is not None:
        try:
            return str(int(retry_after_ms) / 1000)
        except Exception:
            return str(retry_after_ms)

    return "?"

def rate_limit_message(
    error: RateLimitError,
    node: str,
    payload: Dict[str, Any] | None = None,
) -> str:
    """
    Warining message for rate limit exceeded

    Args:
        error (RateLimitError): The rate limit error from OpenAI
        node (str): The node name where the rate limit occurred
        payload (Dict[str, Any] | None): Optional payload data related to the request
    Returns:
        str: The formatted rate limit message with rate limit detail information
    Customs:
        title: "Untitled (Rate Limit Exceeded)"
    """
    # Logging
    error_type = type(error).__name__
    logger.warning("[%s] RATE LIMIT ERROR: %s/%s", node, error_type, error)

    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (Rate Limit Exceeded)"})
    
    # Create Message
    resp = getattr(error, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        headers = dict(headers)
    except Exception:
        headers = {}

    # Extract Rate Limit Info
    remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
    limit_tokens = headers.get("x-ratelimit-limit-tokens")
    retry_after = _get_retry_after(headers)
    remaining_tokens = remaining_tokens if remaining_tokens is not None else "?"
    limit_tokens = limit_tokens if limit_tokens is not None else "?"

    # Create Message
    return RATE_LIMIT_MESSAGE.format(
        remaining_tokens=remaining_tokens,
        limit_tokens=limit_tokens,
        retry_after=retry_after,
    )
