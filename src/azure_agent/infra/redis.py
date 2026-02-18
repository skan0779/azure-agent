from __future__ import annotations

import ipaddress
import inspect

from azure.keyvault.secrets import SecretClient
from redis.asyncio.cluster import RedisCluster


def create_redis_stream_client(secret_client: SecretClient) -> RedisCluster:
    """
    Create Redis stream client for Azure OSS/Enterprise cluster.
    Args:
        secret_client (SecretClient): Azure Key Vault SecretClient instance
    Returns:
        RedisCluster: Asynchronous Redis cluster client instance
    """
    redis_host = str(secret_client.get_secret("REDIS-STREAM-HOST").value)

    def _address_remap(addr: tuple[str, int]) -> tuple[str, int]:
        host, port = addr
        try:
            ipaddress.ip_address(host)
            return redis_host, port
        except ValueError:
            return host, port

    return RedisCluster(
        host=redis_host,
        port=int(secret_client.get_secret("REDIS-STREAM-PORT").value or 10000),
        username=str(secret_client.get_secret("REDIS-STREAM-USERNAME").value),
        password=str(secret_client.get_secret("REDIS-STREAM-ACCESS-KEY").value),
        decode_responses=True,
        ssl=True,
        socket_connect_timeout=5,
        socket_timeout=20,
        address_remap=_address_remap,
    )


async def close_redis_client(redis_client: RedisCluster | None) -> None:
    """
    Close redis client and ConnectionPool.
    Args:
        redis_client (RedisCluster | None): Asynchronous Redis Client Instance
    Returns:
        None
    """
    if redis_client is None:
        return

    try:
        aclose = getattr(redis_client, "aclose", None)
        close = getattr(redis_client, "close", None)
        if callable(aclose):
            await aclose()
        elif callable(close):
            maybe = close()
            if inspect.isawaitable(maybe):
                await maybe
    except Exception:
        pass

    try:
        pool = getattr(redis_client, "connection_pool", None)
        if pool is None:
            return
        disconnect = getattr(pool, "disconnect", None)
        if callable(disconnect):
            maybe = disconnect()
            if inspect.isawaitable(maybe):
                await maybe
    except Exception:
        pass

