import json
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

from config import load_config

# Load Schema
SCHEMA_PATH = Path(__file__).with_name("index_schema.json")


# Helper Function (loading schema)
def load_index_schema(path: Path, index_name: str) -> SearchIndex:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    schema["name"] = index_name
    return SearchIndex.from_dict(schema)


# Create index
def main() -> None:
    config = load_config()

    # Load Search Client
    client = SearchIndexClient(
        endpoint=config.endpoint,
        credential=AzureKeyCredential(config.api_key),
        api_version=config.api_version,
    )

    # Load Index Schema
    index = load_index_schema(SCHEMA_PATH, config.index_name)

    # Create Index
    result = client.create_or_update_index(index)

    # Logging
    print(f"Index ready: {result.name}")


# Run the main function
if __name__ == "__main__":
    main()
