<h1 align="center">Azure AI Search (sample)</h1>

<p align="center">
  Sample dataset and code to create and upload an index document in Azure AI Search from a local PC
</p>

---

## 1. Create Index Schema
> create Azure AI Search index skeleton based on `index_schema.json` and configure `.env.dev` file.

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

## 2. Create Index Documents 
> upload documents to the Azure AI Search index using `index_documents.jsonl`. `index_documents.jsonl` is generated from `dataset.xlsx` example data, customize your own dataset to match `index_schema.json` index schema. 

```bash
uv run python examples/azure_ai_search/create_dcoument.py
```

[Index Document Example](./index_documents.jsonl)
[Upload Document Example](./create_document.py)

---

## 3. Custom Azure AI Search Retriever (optional)
> custom azure ai search wrapper for agent tools. this repository uses simple langchain wrapper, [example](/src/azure_agent/tools/azure_ai_search.py)

[custom azure_ai_search tool example](./azure_ai_search.py)
