
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Tuple

from eaigle.transport.redis_client import RedisClient

class StreamConsumer:
    def __init__(
        self,
        redis: RedisClient,
        streams: List[str],
        group_name: str,
        consumer_name: str,
        block_ms: int = 1000,
        batch_size: int = 10,
    ):
        self._r = redis.raw
        self._streams = streams
        self._group = group_name
        self._consumer = consumer_name
        self._block_ms = block_ms
        self._batch_size = batch_size

    async def setup_groups(self) -> None:
        
        from eaigle.transport.stream_producer import StreamProducer

        class _FakeRedis:
            raw = self._r
        producer = StreamProducer(_FakeRedis())
        for stream in self._streams:
            await producer.ensure_group(stream, self._group)

    async def iter_messages(self) -> AsyncIterator[Tuple[str, str, dict]]:
        
        stream_ids = {s: ">" for s in self._streams}

        while True:
            try:
                results = await self._r.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams=stream_ids,
                    count=self._batch_size,
                    block=self._block_ms,
                )
            except Exception:
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for stream_name, messages in results:
                for msg_id, fields in messages:
                    yield stream_name, msg_id, fields

    async def ack(self, stream_name: str, msg_id: str) -> None:
        await self._r.xack(stream_name, self._group, msg_id)

    async def get_stream_lag(self, stream_name: str) -> int:
        
        info = await self._r.xpending(stream_name, self._group)
        return info.get("pending", 0) if isinstance(info, dict) else 0
