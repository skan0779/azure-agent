# Azure AI Search (examples)
> Example code for create and upload index in Azure AI Search from local PC

---

## 1. Create Index Schema
> create Azure AI Search index skeleton based on **index_schema.json**, 
> configure **.env.dev** file

```bash
python examples/azure_ai_search/create_index.py
```

---


## 2. Create Index Documents 
> upload documents to the Azure AI Search index using **index_documents.jsonl**, 
> generated **index_documents.jsonl** from **dataset.xlsx** data, 
> build **index_documents.jsonl** JSONL file to match the **index_schema.json** settings

```bash
python examples/azure_ai_search/create_document.py
```
