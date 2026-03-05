
from __future__ import annotations

import redis.asyncio as aioredis

class RedisClient:
    def __init__(self, client: aioredis.Redis):
        self._client = client

    @classmethod
    async def create(cls, url: str = "redis://localhost:6379/0") -> "RedisClient":
        client = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        await client.ping()
        return cls(client)

    @property
    def raw(self) -> aioredis.Redis:
        return self._client

    async def close(self) -> None:
        await self._client.aclose()
