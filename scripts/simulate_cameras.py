from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import logging
import sys
import time
import uuid
from pathlib import Path

# Make the src package importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os

import cv2
import numpy as np
import redis.asyncio as aioredis

from eaigle.models.frame import FrameMetadata
from eaigle.preprocessing.shm_utils import write_frame_to_shm
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_producer import StreamProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulator")

# Camera palette: each simulated camera gets a distinct background hue
_PALETTE = [
    (200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50),
    (200, 50, 200), (50, 200, 200), (150, 100, 50), (100, 50, 150),
]


def _generate_frame(
    camera_idx: int,
    seq: int,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """
    Produces a BGR frame with:
      - A base colour unique per camera
      - Gaussian noise to simulate real camera noise
      - A timestamp + camera label burned in
    """
    # Try to load a real sample image so YOLO can find actual objects.
    # Falls back to synthetic frame if no image is available.
    _sample_paths = [
        "/tmp/sample_frame.jpg",
        "/tmp/sample_frame.png",
    ]
    base_frame = None
    for p in _sample_paths:
        if os.path.exists(p):
            loaded = cv2.imread(p)
            if loaded is not None:
                base_frame = cv2.resize(loaded, (width, height))
                break

    if base_frame is not None:
        frame = base_frame.copy()
    else:
        base_color = _PALETTE[camera_idx % len(_PALETTE)]
        frame = np.full((height, width, 3), base_color, dtype=np.uint8)
        noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Overlay timestamp + camera info
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    label = f"CAM_{camera_idx+1:02d} | seq={seq} | {ts_str}"
    cv2.putText(
        frame, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return frame


def _write_frame_sync(camera_idx: int, camera_id: str, seq: int, width: int, height: int):
    """Generates and writes one frame to shm. Runs in a thread executor."""
    frame = _generate_frame(camera_idx, seq, width, height)
    frame_id = str(uuid.uuid4())
    shm_key = write_frame_to_shm(frame_id, frame)
    meta = FrameMetadata.create(
        camera_id=camera_id,
        sequence_num=seq,
        shm_key=shm_key,
        shape=frame.shape,
        dtype=str(frame.dtype),
        pipeline_stage="raw",
    )
    meta.frame_id = frame_id   # override to match shm key
    return meta


async def run_camera(
    camera_idx: int,
    camera_id: str,
    producer: StreamProducer,
    executor: concurrent.futures.ThreadPoolExecutor,
    fps: int,
    duration_s: float,
    stop_event: asyncio.Event,
    width: int = 640,
    height: int = 480,
) -> None:
    interval = 1.0 / fps
    seq = 0
    end_time = time.monotonic() + duration_s if duration_s > 0 else float("inf")
    loop = asyncio.get_running_loop()

    logger.info("Simulator camera %s starting at %d FPS", camera_id, fps)

    while not stop_event.is_set() and time.monotonic() < end_time:
        t0 = loop.time()

        try:
            meta = await loop.run_in_executor(
                executor, _write_frame_sync, camera_idx, camera_id, seq, width, height
            )
            await producer.publish(
                stream_name=f"raw_frames:{camera_id}",
                message=meta.to_dict(),
            )
            seq += 1
        except Exception as exc:
            logger.error("Camera %s error: %s", camera_id, exc)

        elapsed = loop.time() - t0
        sleep_t = max(0.0, interval - elapsed)
        await asyncio.sleep(sleep_t)

    logger.info("Simulator camera %s done (sent %d frames)", camera_id, seq)


async def main(args: argparse.Namespace) -> None:
    redis = await RedisClient.create(args.redis_url)
    producer = StreamProducer(redis)

    # Pre-create consumer groups so the pipeline workers can start
    group = "preprocessing_workers"
    for i in range(args.cameras):
        cam_id = f"cam_{i+1:02d}"
        await producer.ensure_group(f"raw_frames:{cam_id}", group)

    stop_event = asyncio.Event()

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.cameras, 32),
        thread_name_prefix="sim_cam",
    )

    tasks = [
        asyncio.create_task(
            run_camera(
                camera_idx=i,
                camera_id=f"cam_{i+1:02d}",
                producer=producer,
                executor=executor,
                fps=args.fps,
                duration_s=args.duration,
                stop_event=stop_event,
                width=args.width,
                height=args.height,
            )
        )
        for i in range(args.cameras)
    ]

    logger.info(
        "Simulating %d camera(s) at %d FPS for %s",
        args.cameras, args.fps,
        f"{args.duration}s" if args.duration > 0 else "unlimited",
    )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        executor.shutdown(wait=False)
        await redis.close()
        logger.info("Simulator stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock RTSP camera simulator")
    parser.add_argument("--cameras",   type=int,   default=3,     help="Number of cameras to simulate")
    parser.add_argument("--fps",       type=int,   default=10,    help="Frames per second per camera")
    parser.add_argument("--duration",  type=float, default=60.0,  help="Run duration in seconds (0=infinite)")
    parser.add_argument("--width",     type=int,   default=640,   help="Frame width")
    parser.add_argument("--height",    type=int,   default=480,   help="Frame height")
    parser.add_argument("--redis-url", type=str,   default="redis://localhost:6379/0")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
