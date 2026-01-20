<p align="center">
  <img src="docs/icons/Azure-AI-Search.svg" height="72" alt="Azure AI Search" />
</p>

<h1 align="center">Azure AI Search (examples)</h1>

<p align="center">
  Example code to create and upload an index in Azure AI Search from a local PC
</p>

---

## 1. Create Index Schema
> create Azure AI Search index skeleton based on **index_schema.json** and configure **.env.dev** file

```bash
python examples/azure_ai_search/create_index.py
```

---


## 2. Create Index Documents 
> upload documents to the Azure AI Search index using **index_documents.jsonl**

> **index_documents.jsonl** is generated from **dataset.xlsx** data, customize your JSONL file to match **index_schema.json** index schema

```bash
python examples/azure_ai_search/create_document.py
```
