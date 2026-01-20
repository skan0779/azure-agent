from typing import Literal

from pydantic import BaseModel, Field

class AgentType(BaseModel):
    agent_type: Literal["main_agent", "deep_agent"] = Field(..., description="Target agent for the user query.")
