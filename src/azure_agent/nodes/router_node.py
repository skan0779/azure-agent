import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError

from langgraph.config import get_stream_writer

from openai import BadRequestError, RateLimitError, OpenAIError

from messages import parser_error_message, refusal_message, bad_request_error_message, rate_limit_message, default_message

from schemas.state import AgentState
from schemas.router import AgentType

logger = logging.getLogger(__name__)

def create_router_node(
    *, 
    model: AzureChatOpenAI, 
    system_prompt: str
):
    """
    Create a node that performs:
        Agent Type Routing based on user query
        
    Args:
        model: AzureChatOpenAI model instance
        system_prompt: System prompt template for the router
    Raises:
        OutputParserException: If there is an error parsing the model output.
        OpenAIRefusalError: If the model refuses to generate a response.
        BadRequestError: If there is a bad request error from OpenAI.
        RateLimitError: If the rate limit is exceeded.
        OpenAIError: For other OpenAI related errors.
        Exception: For any other exceptions that may occur.
    Messages:
        parser_error_message (OutputParserException)
        bad_request_error_message (BadRequestError)
        rate_limit_message (RateLimitError)
        refusal_message (OpenAIRefusalError)
        default_message (OpenAIError, Exception)
    Customs:
        event: "routing..."
    """

    NODE = "router_node"

    async def router_node(state: AgentState, **_) -> AgentState:

        # System Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_query}"),
            ]
        )

        # Prompt Inputs
        inputs = {
            "system_prompt": system_prompt,
            "user_query": state.get("user_query"),
        }

        # Model Chain
        chain = prompt | model.with_structured_output(AgentType)

        # Invoke Model
        try:
            # Event Streaming
            writer = get_stream_writer()
            writer({"type": "event", "content": "routing..."})

            # Model Invocation
            result = await chain.ainvoke(inputs)
            return {"guardrail": False, "agent_type": result.agent_type}
        
        # Model Output Parsing Error Handling
        except OutputParserException as e:
            msg = parser_error_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}

        # Model Refusal Error Handling
        except OpenAIRefusalError as e:
            msg = refusal_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}
        
        # Bad Request Error Handling
        except BadRequestError as e:
            msg = bad_request_error_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}
        
        # Token Limit Exceeded Error Handling
        except RateLimitError as e:
            msg = rate_limit_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}
        
        # OpenAI Error Handling
        except OpenAIError as e:
            msg = default_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}
        
        # Other Exception Error Handling
        except Exception as e:
            msg = default_message(e, NODE)
            return {"guardrail": True, "messages": [AIMessage(content=msg, additional_kwargs={"stream": True})]}

    return router_node
