from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from langchain_azure_dynamic_sessions import SessionsBashTool

def create_bash_tool(pool_management_endpoint: str):

    @tool
    def bash_tool(
        input: str,
        runtime: ToolRuntime,
    ) -> str:
        """Run bash in an Azure Container Apps Dynamic Session."""
        session_id = f"sandbox-{runtime.context.thread_id}"
        endpoint = pool_management_endpoint.rstrip("/") + "/"

        tool = SessionsBashTool(
            sanitize_input=True,
            pool_management_endpoint=endpoint,
            session_id=session_id,
        )

        return tool.invoke(input)

    return bash_tool
