from langgraph.graph import END

from schemas.state import AgentState

import logging

logger = logging.getLogger(__name__)

def guardrail_conditional_edge(state: AgentState) -> str:
    """
    Conditional Edge from "guardrail" node

    Returns:
        str: name of next node ("router", END)
    """

    # Guardrail Check
    guardrail = state.get("guardrail", False)
    if guardrail:
        logger.info("[guardrail_conditional_edge] : END")
        return END
    else:
        logger.info("[guardrail_conditional_edge] : router")
        return "router"