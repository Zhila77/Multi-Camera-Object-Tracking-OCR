

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import Optional
import cv2
from eaigle.models.frame import FrameMetadata
from eaigle.models.hypothesis import CameraConfig
from eaigle.preprocessing.shm_utils import write_frame_to_shm
from eaigle.transport.stream_producer import StreamProducer

logger = logging.getLogger(__name__)

BACKPRESSURE_THRESHOLD = 300

class CameraWorker:
    

    def __init__(
        self,
        config: CameraConfig,
        producer: StreamProducer,
        executor: concurrent.futures.ThreadPoolExecutor,
        stop_event: asyncio.Event,
    ):
        self.config = config
        self._producer = producer
        self._executor = executor
        self._stop = stop_event
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq: int = 0
        self._frame_interval: float = 1.0 / config.target_fps

    async def run(self) -> None:
        logger.info("Camera %s starting", self.config.camera_id)
        loop = asyncio.get_running_loop()
        consecutive_failures = 0

        while not self._stop.is_set():

            if self._cap is None or not self._cap.isOpened():
                connected = await loop.run_in_executor(
                    self._executor, self._connect
                )
                if not connected:
                    logger.warning(
                        "Camera %s connection failed, retrying in %.1fs",
                        self.config.camera_id,
                        self.config.reconnect_delay_s,
                    )
                    await asyncio.sleep(self.config.reconnect_delay_s)
                    continue
                logger.info("Camera %s connected", self.config.camera_id)
                consecutive_failures = 0

            frame_start = loop.time()

            ret, frame = await loop.run_in_executor(
                self._executor, self._read_frame
            )

            if not ret or frame is None:
                consecutive_failures += 1
                logger.debug(
                    "Camera %s read failure %d/%d",
                    self.config.camera_id,
                    consecutive_failures,
                    self.config.max_consecutive_failures,
                )
                if consecutive_failures >= self.config.max_consecutive_failures:
                    logger.warning(
                        "Camera %s: max failures reached, reconnecting",
                        self.config.camera_id,
                    )
                    await loop.run_in_executor(self._executor, self._release)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0

            meta = FrameMetadata.create(
                camera_id=self.config.camera_id,
                sequence_num=self._seq,
                shm_key="",
                shape=frame.shape,
                dtype=str(frame.dtype),
                pipeline_stage="raw",
            )
            try:
                shm_key = await loop.run_in_executor(
                    self._executor, write_frame_to_shm, meta.frame_id, frame
                )
                meta.shm_key = shm_key
            except Exception as exc:
                logger.error("Camera %s shm write error: %s", self.config.camera_id, exc)
                continue

            try:
                await self._producer.publish(
                    stream_name=f"raw_frames:{self.config.camera_id}",
                    message=meta.to_dict(),
                )
                self._seq += 1
            except Exception as exc:
                logger.error("Camera %s publish error: %s", self.config.camera_id, exc)

            elapsed = loop.time() - frame_start
            sleep_time = max(0.0, self._frame_interval - elapsed)
            if sleep_time:
                await asyncio.sleep(sleep_time)

        await loop.run_in_executor(self._executor, self._release)
        logger.info("Camera %s stopped", self.config.camera_id)

    def _connect(self) -> bool:
        self._release()
        cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        if self.config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap = cap
        return cap.isOpened()

    def _read_frame(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
