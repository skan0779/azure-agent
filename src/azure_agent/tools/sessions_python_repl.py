from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

from azure_agent.graphs.schema import AgentContext

def create_sessions_python_repl_tool(pool_management_endpoint: str):

    @tool
    def sessions_python_repl_tool(
        input: str,
        runtime: ToolRuntime[AgentContext],
    ) -> str:
        """Run Python analysis in an Azure Container Apps Dynamic Session."""
        session_id = f"analyst-{runtime.context.thread_id}"
        endpoint = pool_management_endpoint.rstrip("/") + "/"

        tool = SessionsPythonREPLTool(
            sanitize_input=True,
            pool_management_endpoint=endpoint,
            session_id=session_id,
        )

        return tool.invoke(input)

    return sessions_python_repl_tool