from pydantic import BaseModel, Field, ConfigDict


class PingResponse(BaseModel):
    ping: bool = True


class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="thread/session ID")
    user_id: str = Field(..., description="user ID")
    user_query: str = Field(..., description="user query")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": "44dc72d6-7ba4-44e0-b8e8-0ba2fcb888a6",
                "user_id": "1015520",
                "user_query": "Hello?",
            }
        }
    )


class DeleteThreadRequest(BaseModel):
    thread_id: str = Field(..., description="target Thread ID")


class DeleteThreadResponse(BaseModel):
    thread_id: str = Field(..., description="deleted Thread ID")
    checkpoints: int = 0
    checkpoint_writes: int = 0
    write_registries: int = 0