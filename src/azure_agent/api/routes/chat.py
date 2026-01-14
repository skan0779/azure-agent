import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from utils.sse import sse_pack

from schemas.api import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/agent/api/user_query/stream",
    tags=["Chat"],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string", "content": "string"},
                    "examples": {
                        "delta": {
                            "summary": "delta",
                            "value": 'data: {"type":"delta", "content":"hi there..."}',
                        },
                        "event": {
                            "summary": "event",
                            "value": 'data: {"type":"event_*", "content":"in progress..."}',
                        },
                        "title": {
                            "summary": "title",
                            "value": 'data: {"type":"title", "content":"Title of Session"}',
                        },
                        "error": {
                            "summary": "error",
                            "value": 'data: {"type":"error", "content":"Agent Server Error"}',
                        },
                        "complete": {
                            "summary": "complete",
                            "value": 'data: {"type":"complete"}',
                        },
                    },
                }
            },
        },
    },
)
async def chat_stream(req: ChatRequest, request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=500, detail="Failed Agent Initialization")

    async def gen() -> AsyncGenerator[str, None]:
        try:
            async for evt in agent.main(
                thread_id=req.thread_id,
                user_id=req.user_id,
                user_query=req.user_query,
            ):
                if isinstance(evt, tuple):
                    msg, _meta = evt
                    delta = getattr(msg, "content", "") or ""
                    if delta:
                        yield sse_pack({"type": "delta", "content": delta})
                    continue

                etype = evt.get("type", "")

                if etype == "delta":
                    yield sse_pack({"type": "delta", "content": evt.get("content", "")})
                elif etype == "event":
                    yield sse_pack({"type": "event", "content": evt.get("content", "")})
                elif etype == "title":
                    yield sse_pack({"type": "title", "content": evt.get("content", "")})
                elif etype == "complete":
                    yield sse_pack({"type": "complete"})
                    return

        except Exception as exc:
            logger.warning("[chat.py] Failed to chat_stream: %s", exc)
            yield sse_pack({"type": "error", "content": "Agent Server Error"})
            yield sse_pack({"type": "complete"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
