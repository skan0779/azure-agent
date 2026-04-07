<p align="center">
  <img src="/docs/icons/Azure-AI-Search.svg" height="72" alt="Azure AI Search" />
</p>

<h1 align="center">Azure AI Search (sample)</h1>

<p align="center">
  Sample code to Create and Upload an index document in Azure AI Search from a local PC
</p>

---

## 1. Create Index Schema
> create Azure AI Search index skeleton based on `index_schema.json` and configure `.env.dev` file

[Index Schema Example](./index_schema.json)
[Upload Schema Example](./create_index.py)

Guideline:
- Keep the vector field name as `content_vector` when using LangChain `AzureSearch`.
- The sample app under `src/azure_agent` now relies on LangChain's default field mapping, so you do not need extra `AZURESEARCH_FIELDS_*` overrides if your index follows this schema.
- If you already have an index that uses `embedding` as the vector field name, recreate the index with `content_vector` or update your ingestion data before uploading.

---

## 2. Create Index Documents 
> upload documents to the Azure AI Search index using `index_documents.jsonl`. `index_documents.jsonl` is generated from `dataset.xlsx` data, customize your JSONL file to match `index_schema.json` index schema

[Index Document Example](./index_documents.jsonl)
[Upload Document Example](./create_document.py)

Field mapping:
- `id`: document key
- `title`: document title
- `sub_title`: optional keyword/title support field for semantic config
- `content`: main searchable text
- `page`: page number or chunk order
- `url`: source document URL
- `content_vector`: embedding vector used for vector and hybrid retrieval

Important:
- Every document in `index_documents.jsonl` must use `content_vector` as the embedding field name.
- The field names in the JSONL file must match `index_schema.json` exactly before running [`create_document.py`](./create_document.py).


---

## 3. Custom Azure AI Search Retriever (optional)
> custom azure ai search wrapper for agent tools

[custom azure_ai_search tool example](./azure_ai_search.py)
