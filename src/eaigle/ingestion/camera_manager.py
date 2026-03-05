

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Dict, List
from eaigle.models.hypothesis import CameraConfig
from eaigle.ingestion.camera_worker import CameraWorker
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_producer import StreamProducer
logger = logging.getLogger(__name__)
PREPROCESSING_GROUP = "preprocessing_workers"

class CameraStreamManager:
    def __init__(
        self,
        camera_configs: List[CameraConfig],
        redis: RedisClient,
        stop_event: asyncio.Event,
        max_thread_workers: int = 60,
    ):
        self._configs = camera_configs
        self._redis = redis
        self._stop = stop_event
        self._producer = StreamProducer(redis)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_thread_workers,
            thread_name_prefix="rtsp_reader",
        )
        self._tasks: Dict[str, asyncio.Task] = {}
        self._status: Dict[str, str] = {}

    async def run(self) -> None:

        await self._setup_groups()

        for cfg in self._configs:
            await self._start_camera(cfg)

        while not self._stop.is_set():
            for cfg in self._configs:
                task = self._tasks.get(cfg.camera_id)
                if task is not None and task.done():
                    exc = task.exception()
                    if exc:
                        logger.error(
                            "Camera %s task crashed (%s), restarting",
                            cfg.camera_id, exc,
                        )
                    await self._start_camera(cfg)
            await asyncio.sleep(2)

        logger.info("CameraStreamManager shutting down")
        self._stop.set()
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._executor.shutdown(wait=False)

    async def _start_camera(self, cfg: CameraConfig) -> None:
        worker = CameraWorker(
            config=cfg,
            producer=self._producer,
            executor=self._executor,
            stop_event=self._stop,
        )
        task = asyncio.create_task(
            worker.run(), name=f"camera_{cfg.camera_id}"
        )
        self._tasks[cfg.camera_id] = task
        self._status[cfg.camera_id] = "running"
        logger.info("Started camera task: %s", cfg.camera_id)

    async def _setup_groups(self) -> None:
        for cfg in self._configs:
            stream = f"raw_frames:{cfg.camera_id}"
            await self._producer.ensure_group(stream, PREPROCESSING_GROUP)
        logger.info("Consumer groups ready for %d cameras", len(self._configs))

    def get_status(self) -> Dict[str, str]:
        for cam_id, task in self._tasks.items():
            if task.done():
                self._status[cam_id] = "stopped"
        return dict(self._status)
