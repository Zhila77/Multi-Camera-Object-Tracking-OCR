"""
Frame data models — lightweight descriptors that travel through Redis Streams.
Pixel data stays in POSIX shared memory; only the shm_key pointer is transported.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class FrameMetadata:
    """
    Lightweight descriptor published to Redis Stream.
    No pixel data — only a pointer (shm_key) to POSIX shared memory.
    """
    frame_id: str
    camera_id: str
    capture_ts: float          # Unix timestamp at capture
    sequence_num: int          # Monotonic counter per camera
    shm_key: str               # POSIX shm name: "eaigle_{frame_id}"
    width: int
    height: int
    channels: int              # 3 for BGR/RGB
    dtype: str                 # "uint8" | "float32"
    pipeline_stage: str        # "raw" | "preprocessed"

    @classmethod
    def create(
        cls,
        camera_id: str,
        sequence_num: int,
        shm_key: str,
        shape: Tuple[int, int, int],
        dtype: str = "uint8",
        pipeline_stage: str = "raw",
    ) -> "FrameMetadata":
        h, w, c = shape
        return cls(
            frame_id=str(uuid.uuid4()),
            camera_id=camera_id,
            capture_ts=time.time(),
            sequence_num=sequence_num,
            shm_key=shm_key,
            width=w,
            height=h,
            channels=c,
            dtype=dtype,
            pipeline_stage=pipeline_stage,
        )

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "capture_ts": str(self.capture_ts),
            "sequence_num": str(self.sequence_num),
            "shm_key": self.shm_key,
            "width": str(self.width),
            "height": str(self.height),
            "channels": str(self.channels),
            "dtype": self.dtype,
            "pipeline_stage": self.pipeline_stage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameMetadata":
        return cls(
            frame_id=d["frame_id"],
            camera_id=d["camera_id"],
            capture_ts=float(d["capture_ts"]),
            sequence_num=int(d["sequence_num"]),
            shm_key=d["shm_key"],
            width=int(d["width"]),
            height=int(d["height"]),
            channels=int(d["channels"]),
            dtype=d["dtype"],
            pipeline_stage=d["pipeline_stage"],
        )
