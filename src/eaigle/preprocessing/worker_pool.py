
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from functools import partial
from typing import List

import numpy as np

from eaigle.models.frame import FrameMetadata
from eaigle.preprocessing.pipeline import PreprocessingPipeline
from eaigle.preprocessing.shm_utils import read_frame_from_shm, write_frame_to_shm
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_consumer import StreamConsumer
from eaigle.transport.stream_producer import StreamProducer

logger = logging.getLogger(__name__)

PREPROCESSED_STREAM = "preprocessed_frames"
INFERENCE_GROUP = "inference_workers"

_pipeline: PreprocessingPipeline | None = None

def _pool_init(pipeline_cfg: dict) -> None:
    global _pipeline
    _pipeline = PreprocessingPipeline.from_config(pipeline_cfg)

def _process_frame(
    shm_key: str,
    shape: tuple,
    dtype: str,
) -> tuple[np.ndarray, float]:
    
    assert _pipeline is not None, "Pipeline not initialised in worker process"
    t0 = time.monotonic()
    raw = read_frame_from_shm(shm_key, shape, np.dtype(dtype))
    processed = _pipeline.run(raw)
    return processed, (time.monotonic() - t0) * 1000

class PreprocessingWorkerPool:
    def __init__(
        self,
        redis: RedisClient,
        camera_ids: List[str],
        pipeline_cfg: dict,
        num_workers: int = 4,
        consumer_name: str = "pp_worker_0",
    ):
        self._redis = redis
        self._camera_ids = camera_ids
        self._pipeline_cfg = pipeline_cfg
        self._num_workers = num_workers
        self._consumer_name = consumer_name
        self._producer = StreamProducer(redis)

        self._executor: concurrent.futures.ProcessPoolExecutor | None = None

    async def run(self, stop_event: asyncio.Event) -> None:

        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_workers,
            initializer=_pool_init,
            initargs=(self._pipeline_cfg,),
        )

        await self._producer.ensure_group(PREPROCESSED_STREAM, INFERENCE_GROUP)

        streams = [f"raw_frames:{cam_id}" for cam_id in self._camera_ids]
        consumer = StreamConsumer(
            redis=self._redis,
            streams=streams,
            group_name="preprocessing_workers",
            consumer_name=self._consumer_name,
        )
        await consumer.setup_groups()

        loop = asyncio.get_running_loop()
        tasks: list[asyncio.Task] = []

        async for stream_name, msg_id, fields in consumer.iter_messages():
            if stop_event.is_set():
                break

            meta = FrameMetadata.from_dict(fields)
            shape = (meta.height, meta.width, meta.channels)

            task = asyncio.create_task(
                self._handle_frame(loop, meta, shape, consumer, stream_name, msg_id)
            )
            tasks.append(task)

            if len(tasks) >= self._num_workers * 4:
                done, tasks_set = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                tasks = list(tasks_set)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("PreprocessingWorkerPool stopped")

    async def _handle_frame(
        self,
        loop: asyncio.AbstractEventLoop,
        meta: FrameMetadata,
        shape: tuple,
        consumer: StreamConsumer,
        stream_name: str,
        msg_id: str,
    ) -> None:
        try:
            processed, latency_ms = await loop.run_in_executor(
                self._executor,
                partial(_process_frame, meta.shm_key, shape, meta.dtype),
            )
        except Exception as exc:
            logger.error("Preprocessing failed for frame %s: %s", meta.frame_id, exc)
            await consumer.ack(stream_name, msg_id)
            return

        try:
            new_shm_key = write_frame_to_shm(f"pp_{meta.frame_id}", processed)
        except Exception as exc:
            logger.error("shm write failed: %s", exc)
            await consumer.ack(stream_name, msg_id)
            return

        h, w = processed.shape[:2]
        c = processed.shape[2] if processed.ndim == 3 else 1
        meta.shm_key = new_shm_key
        meta.width = w
        meta.height = h
        meta.channels = c
        meta.dtype = str(processed.dtype)
        meta.pipeline_stage = "preprocessed"

        d = meta.to_dict()
        d["preprocess_latency_ms"] = str(latency_ms)

        await self._producer.publish(PREPROCESSED_STREAM, d)
        await consumer.ack(stream_name, msg_id)
        logger.debug(
            "Preprocessed frame %s in %.1fms", meta.frame_id, latency_ms
        )
