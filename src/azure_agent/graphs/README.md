<p align="center">
  <img src="/docs/icons/LangGraph.webp" height="72" alt="LangGraph" />
</p>

<h1 align="center">LangGraph Agent</h1>

<p align="center">
  LangChain & LangGraph based Azure AI Agent
</p>

---

## 1. Features
> LangChain & LangGraph based Stateful Agent

| Category | Implementation |
| --- | --- |
| Framework | LangChain, LangGraph |
| Models | Azure AI foundry(Model-router, GPT-4o-mini, Text-embedding-3-large) |
| Tools | Azure AI Search, Tavily Search, LangMem |
| Memory | Checkpointer(Redis), Store(PostgreSQL), LangMem(PostgreSQL) |
| Context | SummarizationMiddleware(LangChain) |
| Prompt | Azure Blob Storage |
| Guardrail | Azure AI Content Safety, PIIMiddleware(LangChain) |
| Streaming | FastAPI SSE |

---

## 2. SSE Streaming
> Stream agent progress and model output in SSE-friendly events

SSE Payload shapes:
```json
{"type": "event", "content": "Starting chat ...", "event_id":"1743500000000-0"}
{"type": "event", "content": "Invoking model ...", "event_id":"1743500000001-0"}
{"type": "delta", "content": "Hello", "event_id":"1743500000002-0"}
{"type": "updates", "step": "model", "content": {"messages": []}, "event_id":"1743500000003-0"}
{"type": "cancelled", "content": "Job cancelled", "event_id":"1743500000004-0"}
{"type": "error", "content": "Event stream error", "event_id":"1743500000005-0"}
{"type": "complete", "event_id":"1743500000006-0"}
```

---

## 3. Details

### Models
> Core models used by the agent

- `Main model`(`model-router`): routing llm model based on context difficulty
- `Small model`(`gpt-4o-mini`): small llm model for simple tasks
- `Embedding model`(`text-embedding-3-large`): embedding model for vectorize texts

### Tools
> External tools available to the agent

- `Azure AI Search`: hybrid RAG (semantic search + vector search)
- `Tavily Search`: web search
- `LangMem`: manage long-term memory (`manage_memory`, `search_memory`)

### Memories
> Short-term and long-term memory components

Short-term memory:
- `Checkpointer`: `AsyncShallowRedisSaver`

Long-term memory:
- `Store`: `AsyncPostgresStore`
- `LangMem`: managed `AsyncPostgresStore`

### Context Management
> Control context growth during long conversations

- `SummarizationMiddleware`: Triggers when context reaches `20000` tokens (keep `20` messages)

### Prompts
> Prompt loading and fallback strategy at startup

- `Azure Blob Storage`: Downloads the system prompt
- `Cache`: Stores loaded prompts in in-memory `prompt_cache` and skips reloading the same file
- `Local fallback`: Falls back to `src/azure_agent/prompts/` yaml file

### Guardrails
> Safety checks and execution limits for model and tool calls

- `PIIMiddleware`: guardrail and mask `email`, `credit_card` text inputs
- `Content Safety`: guardrail moderation model inputs
