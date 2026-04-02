from __future__ import annotations

import os, logging

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import azure.identity.aio
import azure.keyvault.secrets.aio

logger = logging.getLogger(__name__)


def get_vault_url(vault_url: str | None = None) -> str:
    vault_url = vault_url or os.getenv("KEY_VAULT_URL")
    if not vault_url:
        logger.error("[key_vault.py] Failed to load KEY_VAULT_URL from environment variables")
        raise RuntimeError("[key_vault.py] Failed to load KEY_VAULT_URL from environment variables")
    return vault_url


def create_secret_client(vault_url: str | None = None) -> SecretClient:
    """
    Create Azure Key Vault SecretClient.
    Args:
        vault_url (str | None): Azure Key Vault URL
    Returns:
        SecretClient: Azure Key Vault SecretClient Instance
    """
    credential = DefaultAzureCredential()
    return SecretClient(
        vault_url=get_vault_url(vault_url),
        credential=credential,
    )


def create_async_secret_client(vault_url: str | None = None):
    credential = azure.identity.aio.DefaultAzureCredential()
    client = azure.keyvault.secrets.aio.SecretClient(
        vault_url=get_vault_url(vault_url),
        credential=credential,
    )
    return client, credential
