from langgraph.graph import END

from schemas.state import AgentState

import logging

logger = logging.getLogger(__name__)

def router_conditional_edge(state: AgentState) -> str:
    """
    Conditional Edge from "router" node

    Returns:
        str: name of next node ("main_agent", "deep_agent", END)
    """
    # Guardrail Check
    guardrail = state.get("guardrail", False)
    if guardrail:
        logger.info("[router_conditional_edge] : END")
        return END
    
    # Agent Type Check
    else:
        agent_type = state.get("agent_type", "main_agent")
        if agent_type == "deep_agent":
            logger.info("[router_conditional_edge] : deep_agent")
            return "deep_agent"
        else:
            logger.info("[router_conditional_edge] : main_agent")
            return "main_agent"