from typing import TypedDict, Annotated, Literal

from utils.message import add_messages_capped


class AgentState(TypedDict, total=False):
    """
    Base Schema for Agent State that performs:
        - store information within StateGraph
        - store conversation messages with Message capping

    Args:
        messages (list): List of messages in the conversation
        thread_id (str): Thread ID
        user_id (str): User ID
        user_query (str): User Question
        agent_type (Literal["main_agent", "document_agent"]): Selected agent type
        guardrail (bool): Guardrail check
    """
    # Messages
    messages: Annotated[list, add_messages_capped]

    # Basic Inputs
    thread_id: Annotated[str, "Thread ID"]
    user_id: Annotated[str, "User ID"]
    user_query: Annotated[str, "User Question"]

    # Router
    agent_type: Annotated[Literal["main_agent", "document_agent"], "Selected Agent Type"]
    
    # Guardrail
    guardrail: Annotated[bool, "Quardrail Check"]
