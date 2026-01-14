from __future__ import annotations
import logging
from typing import Iterable

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

PII_MESSAGE = (
    "**PII({labels}) DETECTED.**"
)

def pii_message(
    labels: Iterable[str], 
    node: str,
) -> str:
    """
    Warning message for PII detections

    Args:
        labels (Iterable[str]): The list of detected PII labels
        node (str): The node name where the PII was detected
    Returns:
        str: The formatted PII warning message
    Customs:
        title: "Untitled (PII Detected)"
    """
    # Logging
    label_text = ", ".join(labels)
    logger.warning("[%s] PII DETECTED: %s", node, label_text)

    # Title Generation
    writer = get_stream_writer()
    writer({"type": "title", "content": "Untitled (PII Detected)"})

    # Create Message
    return PII_MESSAGE.format(labels=label_text)
