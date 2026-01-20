import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Load ENV
ENV_PATH = Path(__file__).resolve().parents[2] / "environments/env/.env.dev"
load_dotenv(dotenv_path=ENV_PATH)

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

    # Load Search Client
    search_client = SearchClient(
        endpoint=os.environ.get("AZURE-AI-SEARCH-ENDPOINT"),
        index_name=os.environ.get("AZURE-AI-SEARCH-INDEX-NAME"),
        credential=AzureKeyCredential(os.environ.get("AZURE-AI-SEARCH-API-KEY")),
        api_version=os.environ.get("AZURE-AI-SEARCH-API-VERSION", "2023-11-01"),
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
