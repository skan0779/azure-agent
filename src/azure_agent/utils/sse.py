import json


def sse_pack(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
