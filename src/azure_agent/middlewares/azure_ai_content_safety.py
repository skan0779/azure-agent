from langchain_core.messages import HumanMessage
from langchain_azure_ai.agents.middleware import (
    AzureContentModerationMiddleware,
    AzurePromptShieldMiddleware,
)
from langchain_azure_ai.agents.middleware.content_safety._prompt_shield import PromptShieldInput


def _last_user_text(state):
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    return None


def azure_content_moderation_middleware(endpoint: str, credential: str):
    """
    Build Azure Content Moderation middleware:
    - Apply to user_input text only
    """
    return AzureContentModerationMiddleware(
        endpoint=endpoint,
        credential=credential,
        categories=["Hate", "SelfHarm", "Sexual", "Violence"],
        severity_threshold=3,
        exit_behavior="error",
        apply_to_input=True,
        apply_to_output=False,
    )


def azure_prompt_shield_middleware(endpoint: str, credential: str):
    """
    Build Azure Prompt Shield middleware:
    - Apply to user_input text only
    """
    return AzurePromptShieldMiddleware(
        endpoint=endpoint,
        credential=credential,
        exit_behavior="error",
        apply_to_input=True,
        apply_to_output=False,
        context_extractor=lambda state, runtime: (
            PromptShieldInput(user_prompt=text, documents=[])
            if (text := _last_user_text(state)) else None
        ),
    )