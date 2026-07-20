<h1 align="center">Azure AI Search (sample)</h1>

<p align="center">
  Sample dataset and code to create an Azure AI Search index and upload documents.
</p>

---

## 1. Configure Environment Variables

Recommended setup from the repository root:

```bash
scripts/setup-azure-ai-search.sh --infra-dir environments/infra
```

Manual execution:

Export the runtime environment variables before running the scripts:

```bash
export AZURE_AI_SEARCH_ENDPOINT="<your-azure-ai-search-endpoint>"
export AZURE_AI_SEARCH_API_KEY="<your-azure-ai-search-api-key>"
export AZURE_AI_SEARCH_INDEX_NAME="azure-agent-index"
export AZURE_AI_SEARCH_API_VERSION="2023-11-01"
```

If you already stored these values in Azure Key Vault, you can load them from Key Vault:

```bash
export KEY_VAULT_NAME="<your-key-vault-name>"

export AZURE_AI_SEARCH_ENDPOINT="$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name AZURE-AI-SEARCH-ENDPOINT --query value -o tsv)"
export AZURE_AI_SEARCH_API_KEY="$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name AZURE-AI-SEARCH-API-KEY --query value -o tsv)"
export AZURE_AI_SEARCH_INDEX_NAME="$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name AZURE-AI-SEARCH-INDEX-NAME --query value -o tsv)"
export AZURE_AI_SEARCH_API_VERSION="$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name AZURE-AI-SEARCH-API-VERSION --query value -o tsv)"
```

---

## 2. Create Index Schema
> Create the Azure AI Search index based on `index_schema.json`.

```bash
uv run python examples/azure_ai_search/create_index.py
```

[Index Schema Example](./index_schema.json)
[Upload Schema Example](./create_index.py)

Notes:
- `id`: document key
- `title`: document title
- `sub_title`: optional keyword/title support field for semantic config
- `content`: main searchable text
- `page`: page number or chunk order
- `url`: source document URL
- `content_vector`: embedding vector used for vector and hybrid retrieval (Every document must use `content_vector` as the embedding field name)

---

## 3. Create Index Documents
> Upload documents to the Azure AI Search index using `index_documents.jsonl`. `index_documents.jsonl` is generated from `dataset.xlsx` example data. Customize your own dataset to match `index_schema.json`.

```bash
uv run python examples/azure_ai_search/create_document.py
```

[Index Document Example](./index_documents.jsonl)
[Upload Document Example](./create_document.py)

---

## 4. Custom Azure AI Search Retriever (optional)
> Custom Azure AI Search wrapper for agent tools. This repository uses a simple LangChain wrapper.

[custom azure_ai_search tool example](./azure_ai_search.py)
