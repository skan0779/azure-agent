<table>
  <tr>
    <td width="110" valign="middle" align="center">
      <img src="/docs/icons/Azure-Agent.png" width="90" alt="Azure Agent" />
    </td>
    <td valign="middle">
      <h1 style="margin: 0;">Azure Agent</h1>
      <p style="margin: 0;">
        Production-ready AI Agent Sample optimized for Azure
      </p>
      <p>
        <a href="LICENSE">
          <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
        </a>
        <a href="https://github.com/skan0779/azure-agent/stargazers">
          <img src="https://img.shields.io/github/stars/skan0779/azure-agent?style=flat" alt="Stars" />
        </a>
        <img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white" alt="Python" />
      </p>
    </td>
  </tr>
</table>


## Quickstart

### Setup (uv)
> Use `uv` for dependency management.
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Environment (required)
> `KEY_VAULT_URL` is required; secrets are pulled from Azure Key Vault.
> Check `environments/env/README.md` for the secret value list.
```bash
export KEY_VAULT_URL="<your-azure-key-vault-url>"
```

### Run Locally (optional)
> Start the FastAPI application via the installed package entrypoint.
```bash
# Re-sync after script changes
uv sync

# API
uv run azure-agent-api
```

> Start the background worker.
```bash
uv run azure-agent-worker
```

### Run with Docker (optional)
> Build and run using the deployment compose file.
```bash
docker compose -f environments/deploy/docker-compose.yml up --build
```


---

## Swagger UI
> Swagger UI assets are bundled in `src/azure_agent/api/static` (air-gapped).
```bash
http://<your-azure-container-host>:8001/agent/swagger
```

---

## License
This project is licensed under the MIT License.
See `LICENSE` for details.

## Third-Party Notices
Bundled Swagger UI assets may require separate upstream license notice files during redistribution.
If you update or re-bundle these assets, include all required third-party notices in the distributed package.
