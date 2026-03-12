"""
HypothesisStore — persists final hypotheses to Redis and broadcasts
them via Redis Pub/Sub so downstream consumers (dashboards, alerters,
analytics) can subscribe without polling.

Storage layout:
  HASH  hypothesis:{hypothesis_id}  → all fields (TTL 60s)
  LIST  camera_hypotheses:{camera_id}  → last 100 hypothesis_ids (LPUSH/LTRIM)
  PUBLISH hypotheses  → JSON event for real-time subscribers
"""
from __future__ import annotations

import json
import logging

from eaigle.models.hypothesis import Hypothesis
from eaigle.transport.redis_client import RedisClient

logger = logging.getLogger(__name__)

HYPOTHESIS_TTL_S = 60
CAMERA_HISTORY_LEN = 100
PUBSUB_CHANNEL = "hypotheses"


class HypothesisStore:
    def __init__(self, redis: RedisClient):
        self._r = redis.raw

    async def save(self, h: Hypothesis) -> None:
        data = h.to_dict()
        key = f"hypothesis:{h.hypothesis_id}"

        # Flatten nested dicts to JSON strings for HSET compatibility
        flat = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in data.items()
        }

        pipe = self._r.pipeline(transaction=False)
        pipe.hset(key, mapping=flat)
        pipe.expire(key, HYPOTHESIS_TTL_S)
        pipe.lpush(f"camera_hypotheses:{h.camera_id}", h.hypothesis_id)
        pipe.ltrim(f"camera_hypotheses:{h.camera_id}", 0, CAMERA_HISTORY_LEN - 1)
        pipe.publish(PUBSUB_CHANNEL, json.dumps({
            "hypothesis_id": h.hypothesis_id,
            "camera_id": h.camera_id,
            "event_description": h.event_description,
            "overall_confidence": h.overall_confidence,
            "pipeline_latency_ms": h.pipeline_latency_ms,
        }))
        await pipe.execute()

        logger.info(
            "[%s] %s  (conf=%.2f, latency=%.0fms)",
            h.camera_id,
            h.event_description,
            h.overall_confidence,
            h.pipeline_latency_ms,
        )

    async def get(self, hypothesis_id: str) -> dict | None:
        data = await self._r.hgetall(f"hypothesis:{hypothesis_id}")
        return data or None

    async def latest_for_camera(self, camera_id: str, n: int = 10) -> list[str]:
        return await self._r.lrange(f"camera_hypotheses:{camera_id}", 0, n - 1)
