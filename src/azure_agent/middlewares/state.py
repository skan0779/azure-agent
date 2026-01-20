from __future__ import annotations

from typing import Any, Dict

from langchain.agents.middleware import after_agent

from langgraph.runtime import Runtime

from schemas.state import AgentState


@after_agent
def update_state_after_agent(state: AgentState, _runtime: Runtime) -> None:
    """Custom middleware to update state after agent"""
    # Schema for New State
    new_state: Dict[str, Any] = {
        "messages": state.get("messages", []),
        "thread_id": state.get("thread_id"),
        "user_id": state.get("user_id"),
        "user_query": state.get("user_query"),
        "agent_type": state.get("agent_type", "deep_agent"),
        "guardrail": bool(state.get("guardrail", False)),
    }

    # Remove Unnecessary Keys
    for k in list(state.keys()):
        if k not in {
            "messages",
            "thread_id",
            "user_id",
            "user_query",
            "agent_type",
            "guardrail",
        }:
            del state[k]

    # Update State
    state.update(new_state)