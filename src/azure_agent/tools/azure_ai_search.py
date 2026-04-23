from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.tools import Tool, create_retriever_tool


def create_azure_ai_search_tool(
    azure_ai_search: AzureSearch,
    top_k: int = 3,
    search_type: str = "semantic_hybrid",
) -> Tool:
    retriever = azure_ai_search.as_retriever(
        search_type=search_type,
        k=top_k,
    )

    return create_retriever_tool(
        retriever=retriever,
        name="azure_ai_search",
        description="Retrieve relevant Microsoft Learn and Azure documents.",
    )
