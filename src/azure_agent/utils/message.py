from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage

from langgraph.graph.message import add_messages

MESSAGE_TURN = 20

def add_messages_capped(
    existing: list[BaseMessage],
    new: list[BaseMessage]
) -> list[BaseMessage]:
    """
    Helper Function to Add messages with capping that performs:
        - trim number of messages (MESSAGE_TURN of user messages)
        - cleanup ToolMessage
        - cleanup tool_calls

    Args:
        existing (list[BaseMessage]): Existing messages.
        new (list[BaseMessage]): New messages to add.
    Returns:
        list[BaseMessage]: Merged and trimmed messages.
    """
    # Merge messages
    merged = add_messages(existing, new)
    user_msgs = [m for m in merged if isinstance(m, HumanMessage)]
    trimmed = merged

    # Trim messages
    if len(user_msgs) > MESSAGE_TURN:
        turns = 0
        start_idx = 0
        for i in range(len(merged) - 1, -1, -1):
            m = merged[i]
            if isinstance(m, HumanMessage):
                turns += 1
                if turns == MESSAGE_TURN:
                    start_idx = i
                    break

        trimmed = merged[start_idx:]

    # Cleanup ToolMessage and tool_calls
    last_human_idx = None
    for i, m in enumerate(trimmed):
        if isinstance(m, HumanMessage):
            last_human_idx = i

    if last_human_idx is None:
        return trimmed

    cleaned: list[BaseMessage] = []
    for i, m in enumerate(trimmed):
        if i < last_human_idx:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue
        cleaned.append(m)

    return cleaned