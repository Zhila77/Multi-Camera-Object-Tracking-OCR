
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class FrameMetadata:
    
    frame_id: str
    camera_id: str
    capture_ts: float
    sequence_num: int
    shm_key: str
    width: int
    height: int
    channels: int
    dtype: str
    pipeline_stage: str

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
