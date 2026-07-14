from __future__ import annotations

import ipaddress
import inspect

from redis.asyncio.cluster import RedisCluster

from azure_agent.config import RedisStreamSettings


def create_redis_stream_client(settings: RedisStreamSettings) -> RedisCluster:
    """
    Create Redis stream client for Azure OSS/Enterprise cluster.
    Args:
        settings (RedisStreamSettings): Redis stream connection settings.
    Returns:
        RedisCluster: Asynchronous Redis cluster client instance
    """
    redis_host = settings.host

    def _address_remap(addr: tuple[str, int]) -> tuple[str, int]:
        host, port = addr
        try:
            ipaddress.ip_address(host)
            return redis_host, port
        except ValueError:
            return host, port

    return RedisCluster(
        host=redis_host,
        port=settings.port,
        username=settings.username,
        password=settings.access_key,
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
