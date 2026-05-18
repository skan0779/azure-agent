from dataclasses import dataclass
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel

@dataclass
class AgentContext:
    """
    Runtime context for deep agents.

    This is not automatically injected into the prompt. It exists so the
    runtime, tools, and backends can access per-request identity data.
    """
    
    user_id: str
    thread_id: str
    job_id: str

class AgentState(TypedDict, total=False):
    """
    Base Schema for StateGraph

    Args:
        messages (list): List of messages in the conversation
        thread_id (str): Thread ID
        user_id (str): User ID
        user_query (str): User Question
        guardrail (bool): Guardrail check
    """

    messages: Annotated[list, add_messages]
    thread_id: Annotated[str, "Thread ID"]
    user_id: Annotated[str, "User ID"]
    user_query: Annotated[str, "User Question"]
    guardrail: Annotated[bool, "Guardrail Check"]

class UserProfile(BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    role: Optional[str] = None
    organization: Optional[str] = None
