
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from eaigle.aggregation.hypothesis_fusion import HypothesisFusion
from eaigle.aggregation.hypothesis_store import HypothesisStore
from eaigle.models.detection import StageResult, Detection, BoundingBox
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_consumer import StreamConsumer

logger = logging.getLogger(__name__)

FRAME_TIMEOUT_S = 2.0
EXPECTED_STAGES = {"primary", "secondary"}

@dataclass
class FrameAccumulator:
    frame_id: str
    camera_id: str
    capture_ts: float
    stages: Dict[str, StageResult] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    @property
    def is_complete(self) -> bool:
        return EXPECTED_STAGES.issubset(self.stages.keys())

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.created_at

class ResultCollector:
    def __init__(
        self,
        redis: RedisClient,
        fusion: HypothesisFusion,
        store: HypothesisStore,
        consumer_name: str = "collector_0",
    ):
        self._redis = redis
        self._fusion = fusion
        self._store = store
        self._consumer_name = consumer_name
        self._accum: Dict[str, FrameAccumulator] = {}

    async def run(self, stop_event: asyncio.Event) -> None:
        consumer = StreamConsumer(
            redis=self._redis,
            streams=["inference_results"],
            group_name="aggregation_workers",
            consumer_name=self._consumer_name,
        )
        await consumer.setup_groups()

        cleanup_task = asyncio.create_task(self._cleanup_loop(stop_event))

        async for stream_name, msg_id, fields in consumer.iter_messages():
            if stop_event.is_set():
                break
            try:
                await self._handle_message(fields)
            except Exception as exc:
                logger.error("ResultCollector error: %s", exc)
            await consumer.ack(stream_name, msg_id)

        cleanup_task.cancel()

        for acc in list(self._accum.values()):
            await self._flush(acc)

        logger.info("ResultCollector stopped")

    async def _handle_message(self, fields: dict) -> None:
        capture_ts = float(fields["capture_ts"])
        result_dict = json.loads(fields["result_json"])

        stage_result = _stage_result_from_dict(result_dict)
        frame_id = stage_result.frame_id
        camera_id = stage_result.camera_id

        if frame_id not in self._accum:
            self._accum[frame_id] = FrameAccumulator(
                frame_id=frame_id,
                camera_id=camera_id,
                capture_ts=capture_ts,
            )

        acc = self._accum[frame_id]
        acc.stages[stage_result.stage] = stage_result

        if acc.is_complete:
            await self._flush(acc)
            del self._accum[frame_id]

    async def _flush(self, acc: FrameAccumulator) -> None:
        stage_results = list(acc.stages.values())
        hypothesis = self._fusion.fuse(
            frame_id=acc.frame_id,
            camera_id=acc.camera_id,
            capture_ts=acc.capture_ts,
            stage_results=stage_results,
        )
        await self._store.save(hypothesis)

    async def _cleanup_loop(self, stop_event: asyncio.Event) -> None:
        
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
            timed_out = [
                acc for acc in self._accum.values()
                if acc.age_s > FRAME_TIMEOUT_S
            ]
            for acc in timed_out:
                logger.debug(
                    "Frame %s timed out after %.1fs with stages %s",
                    acc.frame_id, acc.age_s, list(acc.stages.keys()),
                )
                await self._flush(acc)
                self._accum.pop(acc.frame_id, None)

def _stage_result_from_dict(d: dict) -> StageResult:
    detections = []
    for det_d in d.get("detections", []):
        bbox_d = det_d["bbox"]
        det = Detection(
            detection_id=det_d["detection_id"],
            frame_id=det_d["frame_id"],
            camera_id=det_d["camera_id"],
            stage=det_d["stage"],
            label=det_d["label"],
            confidence=det_d["confidence"],
            bbox=BoundingBox(**bbox_d),
            attributes=det_d.get("attributes", {}),
            parent_detection_id=det_d.get("parent_detection_id"),
            ocr_text=det_d.get("ocr_text"),
        )
        detections.append(det)

    return StageResult(
        frame_id=d["frame_id"],
        camera_id=d["camera_id"],
        stage=d["stage"],
        detections=detections,
        inference_latency_ms=d.get("inference_latency_ms", 0.0),
        model_id=d.get("model_id", "unknown"),
    )
