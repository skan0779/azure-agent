from langchain.agents.middleware import before_agent, before_model

from langgraph.runtime import Runtime

from schemas.state import AgentState


@before_agent
async def event_stream_before_agent(state: AgentState, runtime: Runtime) -> None:
    '''Custom middleware to stream events before strating agent'''

    agent_type = state.get("agent_type")
    runtime.stream_writer({"type": "event", "content": f"Starting {agent_type} ..."})


@before_model
async def event_stream_before_model(state: AgentState, runtime: Runtime) -> None:
    '''Custom middleware to stream events before invoking model'''

    runtime.stream_writer({"type": "event", "content": f"Invoking model ..."})
