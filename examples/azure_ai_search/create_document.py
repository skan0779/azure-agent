import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from config import load_config

# Load Data
DATA_PATH = Path(__file__).with_name("index_documents.jsonl")

# Configuration
BATCH_SIZE = 100


# Helper Function (batching)
def _iter_batches(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# Main Function
def main() -> None:
    config = load_config()

    # Load Search Client
    search_client = SearchClient(
        endpoint=config.endpoint,
        index_name=config.index_name,
        credential=AzureKeyCredential(config.api_key),
        api_version=config.api_version,
    )

    # Load JSONL
    documents: List[Dict[str, Any]] = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    if not documents:
        print("Failed: Document Not Found")
        return

    # Upload Documents (Batch)
    job = False
    for batch in _iter_batches(documents, BATCH_SIZE):
        results = search_client.merge_or_upload_documents(documents=batch)
        failures = [r for r in results if not r.succeeded]
        if failures:
            job = True
            for r in failures:
                print("Failed: Document: '%s': %s" % (r.key, r.error_message))
    if job:
        print("Success: Upload with Failure")
    else:
        print(f"Success: Upload Completed: {len(documents)} documents")


if __name__ == "__main__":
    main()
