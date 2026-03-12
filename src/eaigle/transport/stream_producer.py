"""
Redis Streams producer — publishes lightweight frame metadata pointers.
"""
from __future__ import annotations

from eaigle.transport.redis_client import RedisClient

# Cap each stream at 500 messages to bound memory usage.
# At 10 FPS this is 50 seconds of head-room per camera.
STREAM_MAXLEN = 500


class StreamProducer:
    def __init__(self, redis: RedisClient):
        self._r = redis.raw

    async def publish(self, stream_name: str, message: dict) -> str:
        """XADD with approximate trimming. Returns the generated message ID."""
        msg_id = await self._r.xadd(
            stream_name,
            message,
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )
        return msg_id

    async def ensure_group(self, stream_name: str, group_name: str) -> None:
        """Create consumer group idempotently, starting from new messages."""
        try:
            await self._r.xgroup_create(
                stream_name, group_name, id="$", mkstream=True
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
