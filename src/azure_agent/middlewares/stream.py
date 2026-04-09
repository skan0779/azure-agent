from __future__ import annotations

from langchain.agents.middleware import before_agent, before_model

from langgraph.runtime import Runtime

from azure_agent.graphs.schema import AgentState


@before_agent
async def event_stream_before_agent(state: AgentState, runtime: Runtime) -> None:
    '''Custom middleware to stream custom events before starting agent'''

    runtime.stream_writer({"type": "event", "content": f"Starting agent ..."})


@before_model
async def event_stream_before_model(state: AgentState, runtime: Runtime) -> None:
    '''Custom middleware to stream custom events before invoking model'''

    runtime.stream_writer({"type": "event", "content": f"Invoking model ..."})
