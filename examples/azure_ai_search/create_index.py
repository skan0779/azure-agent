import json, os

from pathlib import Path

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

# Load ENV
ENV_PATH = Path(__file__).resolve().parents[2] / "environments/env/.env.dev"
load_dotenv(dotenv_path=ENV_PATH)

# Set ENV
AZURE_AI_SEARCH_ENDPOINT=os.environ.get("AZURE-AI-SEARCH-ENDPOINT")
AZURE_AI_SEARCH_API_KEY=os.environ.get("AZURE-AI-SEARCH-API-KEY")
AZURE_AI_SEARCH_API_VERSION=os.environ.get("AZURE-AI-SEARCH-API-VERSION")

# Load Schema
SCHEMA_PATH = Path(__file__).with_name("index_schema.json")

# Helper Function (loading schema)
def load_index_schema(path: Path) -> SearchIndex:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    return SearchIndex.from_dict(schema)

# Create index
def main() -> None:

    # Load Search Client
    client = SearchIndexClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_SEARCH_API_KEY),
        api_version=AZURE_AI_SEARCH_API_VERSION,
    )

    # Load Index Schema
    index = load_index_schema(SCHEMA_PATH)

    # Create Index
    result = client.create_or_update_index(index)

    # Logging
    print(f"Index ready: {result.name}")

# Run the main function
if __name__ == "__main__":
    main()