from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(TypedDict, total=False):
    """
    Base Schema for StateGraph

    Args:
        messages (list): List of messages in the conversation
        thread_id (str): Thread ID
        user_id (str): User ID
        user_query (str): User Question
        agent_type (Literal["main_agent", "deep_agent"]): Selected agent type
        guardrail (bool): Guardrail check
    """

    messages: Annotated[list, add_messages]
    thread_id: Annotated[str, "Thread ID"]
    user_id: Annotated[str, "User ID"]
    user_query: Annotated[str, "User Question"]
    agent_type: Annotated[Literal["main_agent", "deep_agent"], "Selected Agent Type"]
    guardrail: Annotated[bool, "Guardrail Check"]


class UserProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = "ko"
