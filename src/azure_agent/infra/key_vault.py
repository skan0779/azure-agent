from __future__ import annotations

import os, logging

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)


def create_secret_client(vault_url: str | None = None) -> SecretClient:
    """
    Create Azure Key Vault SecretClient.
    Args:
        vault_url (str | None): Azure Key Vault URL
    Returns:
        SecretClient: Azure Key Vault SecretClient Instance
    """
    vault_url = vault_url or os.getenv("KEY_VAULT_URL")
    if not vault_url:
        logger.error("[key_vault.py] Failed to load KEY_VAULT_URL from environment variables")
        raise RuntimeError("[key_vault.py] Failed to load KEY_VAULT_URL from environment variables")

    credential = DefaultAzureCredential()
    return SecretClient(
        vault_url=vault_url,
        credential=credential,
    )
