<h1 align="center">Token Encoder</h1>

<p align="center">
  Bundled tiktoken cache files used through </code>TIKTOKEN_CACHE_DIR</code>.
  The runtime default is </code>src/azure_agent/encoder</code>.
</p>

![tiktoken 0.11.0](https://img.shields.io/badge/tiktoken-0.11.0-6E4AFF) 
![Air-gapped ready](https://img.shields.io/badge/Air--gapped-ready-0E7C7B)

---

## 1. Encoder list
> tiktoken encoder cache for air-gapped environment

| Encoder | Supported Model | Cache Hash |
| --- | --- | --- |
| **cl100k_base** | gpt-3.5-turbo, gpt-4-turbo, text-embedding-3-small/large | 9b5ad71b2ce5302211f9c61530b329a4922fc6a4 |
| **o200k_base** | gpt-4o, gpt-4.1, gpt-5 | fb374d419588a4632f3f557e76b4b70aebbca790 |

---

## 2. Add new encoder

### 2.1 Create files
> create new encoder cache files
```bash
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache
python - <<'PY'
import tiktoken
tiktoken.get_encoding("<encoder_name>")
PY
ls -l $TIKTOKEN_CACHE_DIR
```

### 2.2 Copy files 
> copy new files to `src/encoder/` folder
