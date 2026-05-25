<p align="center">
  <img src="/docs/icons/Azure-Storage-Account.svg" height="72" alt="Azure Storage Account" />
</p>

<h1 align="center">Azure Storage Account (blob storage)</h1>

<p align="center">
  Manage Prompts from Azure Blob Storage 
</p>

---

## Overview
> `prompts/` contains YAML system prompt files used by the agent runtime.

At startup, the runtime loads prompt files from Azure Blob Storage first.
If Blob loading fails, it falls back to the local file under `src/azure_agent/prompts/`.

Prompt must be valid YAML files:
```yaml
system: |
  You are a helpful assistant.
```

Agent Runtime reads prompts from container configured by: `BLOB_CONNECTION_STRING`


## Notes
Blob is the primary source for production prompt management. 
Local files are fallback only. 
(Prompt content is cached in memory after loading.)
