import logging
from pydantic import BaseModel
from typing import Optional, List

from langchain_core.tools import Tool, StructuredTool
from langchain_openai import AzureOpenAIEmbeddings

from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery, QueryCaptionType, QueryAnswerType

logger = logging.getLogger(__name__)

# Search Parameters
SEARCH_FIELDS = ['content', 'title', 'sub_title']

# Return Parameters
SELECT_FIELDS = ['id', 'content', 'title', 'page', 'url']

class AzureAISearchService:

    def __init__(
        self, 
        azure_ai_search_client: SearchClient, 
        embedding_model: AzureOpenAIEmbeddings, 
        semantic_config: str = "default",
        top_k: int = 3
    ):
        """
        Initalize Azure AI Search Tool Configuration

        Args:
            azure_ai_search_client: Azure AI Search Client Instance
            embedding_model: Azure OpenAI Embedding Model Instance
            semantic_config: Semantic Configuration Name
            top_k: Number of top results to return
        """
        self.azure_ai_search_client = azure_ai_search_client
        self.embedding_model = embedding_model
        self.semantic_config = semantic_config
        self.top_k = top_k

    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Generate Embeddings for Query Text

        Args:
            text (str): input text
        Returns:
            Optional[List[float]]: embedding vector
        Raises:
            Exception: embedding failure
        """
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            logger.warning("[azure_ai_search.py] Failed to embed query : %s", e)           
            return None

    def _serialize(self, result) -> dict:
        """
        Serialize Azure AI Search Result to JSON-serializable dict

        Args:
            result: Azure AI Search Result Object
        Returns:
            dict: JSON-serializable search result
        Raises:
            Exception: serialization failure
        """
        try:
            out = {}
            for k, v in result.items():

                # serialize captions field
                if k == "@search.captions":
                    if v is not None:
                        captions = []
                        for caption in v:
                            try:
                                caption_dict = {
                                    'text': getattr(caption, 'text', ''),
                                    'highlights': getattr(caption, 'highlights', '')
                                }
                                captions.append(caption_dict)
                            except Exception as caption_error:
                                logger.warning("[azure_ai_search.py] Failed to Serialize @search.captions : %s", caption_error)
                                captions.append({'text': str(caption), 'highlights': ''})
                        out[k] = captions
                    else:
                        out[k] = []

                # serialize answers field        
                elif k == "@search.answers":
                    if v is not None:
                        answers = []
                        for answer in v:
                            try:
                                answer_dict = {
                                    'text': getattr(answer, 'text', ''),
                                    'highlights': getattr(answer, 'highlights', ''),
                                    'score': getattr(answer, 'score', None)
                                }
                                answers.append(answer_dict)
                            except Exception as answer_error:
                                logger.warning("[azure_ai_search.py] Failed to Serialize @search.answers : %s", answer_error)
                                answers.append({'text': str(answer), 'highlights': '', 'score': None})
                        out[k] = answers
                    else:
                        out[k] = []

                # serialize general fields
                else:
                    out[k] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)

            # Logging
            # logger.info("[azure_ai_search.py] Search Result : %s, %s, %s", out.get('id'), out.get('title'), out.get('page'))
            return out
        
        # Exception Handling
        except Exception as e:
            logger.warning("[azure_ai_search.py] Failed to Serialize : %s", e)
            return {
                'id': result.get('id', ''),
                'content': str(result.get('content', '')),
                'title': result.get('title', ''),
                'page': result.get('page', 0),
                'url': result.get('url', ''),
                '@search.score': result.get('@search.score', 0.0),
                '@search.highlights': result.get('@search.highlights', None),
                # 'error': f"Serialization error: {str(e)}"
            }

    def search(self, query: str) -> List[dict]:
        """
        Azure AI Search main search function performing hybrid search (semantic, vector)
        
        Args:
            query (str): text qeury
        Returns:
            List[dict]: search results
        Raises:
            Exception: search failure
        """
        # Embedding Query
        embedding = self._embed(query)
        if embedding is None:
            logger.warning("[azure_ai_search.py] Failed to Generate Query Embeddings")
            return []

        # Search Execution
        try:
            # Search Parameters
            search_params = {
                'search_text': query,
                'top': self.top_k,
                'query_type': QueryType.SEMANTIC,
                "semantic_configuration_name": self.semantic_config,
                "query_caption": QueryCaptionType.EXTRACTIVE,
                "query_answer": QueryAnswerType.EXTRACTIVE,
                'select': SELECT_FIELDS,
                'search_fields': SEARCH_FIELDS,
                'vector_queries': [VectorizedQuery(
                    vector=embedding,
                    k_nearest_neighbors=5,
                    fields='embedding'
                )]
            }

            # Search Call
            results = self.azure_ai_search_client.search(**search_params)
            return [self._serialize(r) for r in results]
        
        # Exception Handling
        except Exception as e:
            logger.warning("[azure_ai_search.py] Failed to Search: %s", e)
            return []

class SearchInput(BaseModel):
    """
    Input Schema for azure_ai_search_tool

    Args:
        query (str): search query
    """
    query: str

def create_azure_ai_search_tool(
    azure_ai_search_client: SearchClient,
    embedding_model: AzureOpenAIEmbeddings,
    semantic_config: str = "default",
    top_k: int = 3,
) -> Tool:
    """
    Create Structured Tool for Azure AI Search
    
    Args:
        azure_ai_search_client: Azure AI Search SDK client
        embedding_model: Azure OpenAI embeddings
        semantic_config: semantic configuration name
        top_k: Number of top results to return
    Returns:
        Tool (StructuredTool): azure_ai_search_tool
    """
    # Create Azure AI Search Service
    service = AzureAISearchService(
        azure_ai_search_client=azure_ai_search_client,
        embedding_model=embedding_model,
        semantic_config=semantic_config,
        top_k=top_k,
    )

    # Define Search Function
    def _run(query: str):
        """
        Search function wrapper for StructuredTool

        Args:
            query (str): search query
        Returns:
            List[dict]: search results
        """
        logger.info("[azure_ai_search.py] Search Request : %s", query)
        return service.search(query)
    
    # Create Structured Tool
    return StructuredTool(
        name="azure_ai_search_tool", 
        description="Azure AI Search Tool for semantic and vector search over documents.",
        func=_run,
        args_schema=SearchInput
    )
