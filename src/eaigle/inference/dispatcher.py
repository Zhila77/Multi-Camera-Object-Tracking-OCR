

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import List

import httpx
import numpy as np

from eaigle.inference.batcher import DynamicBatcher
from eaigle.models.detection import BoundingBox, Detection, StageResult
from eaigle.models.frame import FrameMetadata
from eaigle.preprocessing.shm_utils import read_frame_from_shm
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_consumer import StreamConsumer
from eaigle.transport.stream_producer import StreamProducer

logger = logging.getLogger(__name__)

RESULTS_STREAM = "inference_results"
AGGREGATION_GROUP = "aggregation_workers"

class InferenceDispatcher:
    def __init__(
        self,
        redis: RedisClient,
        inference_url: str,
        batch_size: int = 8,
        batch_timeout_ms: float = 50.0,
        consumer_name: str = "dispatcher_0",
        http_timeout_s: float = 5.0,
    ):
        self._redis = redis
        self._inference_url = inference_url.rstrip("/")
        self._batch_size = batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._consumer_name = consumer_name
        self._http_timeout = http_timeout_s
        self._producer = StreamProducer(redis)
        self._client: httpx.AsyncClient | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        await self._producer.ensure_group(RESULTS_STREAM, AGGREGATION_GROUP)

        self._client = httpx.AsyncClient(timeout=self._http_timeout)

        batcher = DynamicBatcher(
            max_size=self._batch_size,
            max_wait_ms=self._batch_timeout_ms,
            flush_callback=self._dispatch_batch,
        )

        consumer = StreamConsumer(
            redis=self._redis,
            streams=["preprocessed_frames"],
            group_name="inference_workers",
            consumer_name=self._consumer_name,
        )
        await consumer.setup_groups()

        async for stream_name, msg_id, fields in consumer.iter_messages():
            if stop_event.is_set():
                break
            meta = FrameMetadata.from_dict(fields)
            await batcher.add(meta)
            await consumer.ack(stream_name, msg_id)

        await batcher.flush_remaining()
        await self._client.aclose()
        logger.info("InferenceDispatcher stopped")

    async def _dispatch_batch(self, batch: List[FrameMetadata]) -> None:
        if not batch:
            return

        batch_id = str(uuid.uuid4())
        t0 = time.monotonic()

        frames_payload = []
        loaded_frames: dict[str, tuple[np.ndarray, FrameMetadata]] = {}

        for meta in batch:
            try:
                shape = (meta.height, meta.width, meta.channels)
                arr = read_frame_from_shm(meta.shm_key, shape, np.dtype(meta.dtype))
                if arr.size == 0:
                    logger.warning("Empty frame %s from shm — skipping", meta.frame_id)
                    continue
                loaded_frames[meta.frame_id] = (arr, meta)
                frames_payload.append({
                    "frame_id": meta.frame_id,
                    "camera_id": meta.camera_id,
                    "data_b64": base64.b64encode(arr.tobytes()).decode(),
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.name,
                    "stage": "primary",
                })
            except Exception as exc:
                logger.warning("Could not load frame %s from shm: %s", meta.frame_id, exc)

        if not frames_payload:
            return

        primary_results = await self._post_inference(
            "/detect/primary", batch_id, frames_payload
        )
        if primary_results is None:
            return

        stage_a_detections: List[Detection] = []
        for det_dict in primary_results.get("results", []):
            bbox_d = det_dict["bbox"]
            det = Detection.create(
                frame_id=det_dict["frame_id"],
                camera_id=batch[0].camera_id,
                stage="primary",
                label=det_dict["label"],
                confidence=det_dict["confidence"],
                bbox=BoundingBox(**bbox_d),
            )

            orig = next(
                (m for m in batch if m.frame_id == det_dict["frame_id"]), None
            )
            if orig:
                det.camera_id = orig.camera_id
            stage_a_detections.append(det)

        for meta in batch:
            frame_dets = [d for d in stage_a_detections if d.frame_id == meta.frame_id]
            result = StageResult(
                frame_id=meta.frame_id,
                camera_id=meta.camera_id,
                stage="primary",
                detections=frame_dets,
                inference_latency_ms=primary_results.get("inference_latency_ms", 0),
                model_id=primary_results.get("model_id", "unknown"),
            )
            await self._publish_result(meta.capture_ts, result)

        crop_payload = []
        for det in stage_a_detections:
            orig_frame_meta = next(
                (m for m in batch if m.frame_id == det.frame_id), None
            )
            if orig_frame_meta is None:
                continue
            arr, _ = loaded_frames.get(det.frame_id, (None, None))
            if arr is None:
                continue

            crop = self._extract_crop(arr, det.bbox)
            if crop.size == 0:
                continue

            crop_payload.append({
                "frame_id": det.frame_id,
                "camera_id": det.camera_id,
                "data_b64": base64.b64encode(crop.tobytes()).decode(),
                "shape": list(crop.shape),
                "dtype": str(crop.dtype),
                "stage": "secondary",
                "parent_detection_id": det.detection_id,
                "label_hint": det.label,
            })

        if crop_payload:
            secondary_results = await self._post_inference(
                "/detect/secondary", batch_id + "_B", crop_payload
            )
            if secondary_results:
                sec_dets: List[Detection] = []
                for det_dict in secondary_results.get("results", []):
                    bbox_d = det_dict["bbox"]
                    det = Detection.create(
                        frame_id=det_dict["frame_id"],
                        camera_id="",
                        stage="secondary",
                        label=det_dict["label"],
                        confidence=det_dict["confidence"],
                        bbox=BoundingBox(**bbox_d),
                        parent_detection_id=det_dict.get("parent_detection_id"),
                        ocr_text=det_dict.get("ocr_text"),
                    )
                    orig = next(
                        (m for m in batch if m.frame_id == det_dict["frame_id"]), None
                    )
                    if orig:
                        det.camera_id = orig.camera_id
                    sec_dets.append(det)

                for meta in batch:
                    frame_dets = [d for d in sec_dets if d.frame_id == meta.frame_id]
                    if frame_dets:
                        result = StageResult(
                            frame_id=meta.frame_id,
                            camera_id=meta.camera_id,
                            stage="secondary",
                            detections=frame_dets,
                            inference_latency_ms=secondary_results.get(
                                "inference_latency_ms", 0
                            ),
                            model_id=secondary_results.get("model_id", "unknown"),
                        )
                        await self._publish_result(meta.capture_ts, result)

        logger.info(
            "Dispatched batch %s: %d frames in %.1fms",
            batch_id, len(batch), (time.monotonic() - t0) * 1000,
        )

    async def _post_inference(
        self, path: str, batch_id: str, frames: list
    ) -> dict | None:
        try:
            resp = await self._client.post(
                self._inference_url + path,
                json={"batch_id": batch_id, "frames": frames},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Inference call to %s failed: %s", path, exc)
            return None

    async def _publish_result(self, capture_ts: float, result: StageResult) -> None:
        payload = {
            "capture_ts": str(capture_ts),
            "result_json": json.dumps(result.to_dict()),
        }
        await self._producer.publish(RESULTS_STREAM, payload)

    @staticmethod
    def _extract_crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.to_pixel_coords(w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2].copy()
