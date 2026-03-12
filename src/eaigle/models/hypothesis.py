"""
Final hypothesis model — the fused, human-readable conclusion produced
by the Data Integrator layer from all inference stage results.

Examples:
  "Vehicle ABC123 detected at Gate 3"
  "Unauthorized person detected in Zone A"
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from eaigle.models.detection import Detection


@dataclass
class Hypothesis:
    """
    The final output of one complete pipeline cycle for a single frame.
    Captures what was seen, where, when, and with what confidence.
    """
    hypothesis_id: str
    frame_id: str
    camera_id: str
    capture_ts: float           # When the frame was captured
    final_ts: float             # When the hypothesis was produced
    primary_detections: List[Detection]     # Stage-1 results (vehicle, person…)
    secondary_detections: List[Detection]   # Stage-2 results (plate, badge…)
    event_description: str      # Human-readable summary
    overall_confidence: float
    fusion_strategy: str
    pipeline_latency_ms: float  # capture_ts → final_ts

    @classmethod
    def create(
        cls,
        frame_id: str,
        camera_id: str,
        capture_ts: float,
        primary_detections: List[Detection],
        secondary_detections: List[Detection],
        event_description: str,
        overall_confidence: float,
        fusion_strategy: str = "confidence_nms_temporal",
    ) -> "Hypothesis":
        final_ts = time.time()
        return cls(
            hypothesis_id=str(uuid.uuid4()),
            frame_id=frame_id,
            camera_id=camera_id,
            capture_ts=capture_ts,
            final_ts=final_ts,
            primary_detections=primary_detections,
            secondary_detections=secondary_detections,
            event_description=event_description,
            overall_confidence=overall_confidence,
            fusion_strategy=fusion_strategy,
            pipeline_latency_ms=(final_ts - capture_ts) * 1000,
        )

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "capture_ts": self.capture_ts,
            "final_ts": self.final_ts,
            "primary_detections": [d.to_dict() for d in self.primary_detections],
            "secondary_detections": [d.to_dict() for d in self.secondary_detections],
            "event_description": self.event_description,
            "overall_confidence": self.overall_confidence,
            "fusion_strategy": self.fusion_strategy,
            "pipeline_latency_ms": self.pipeline_latency_ms,
        }


@dataclass
class CameraConfig:
    camera_id: str
    rtsp_url: str
    target_fps: int = 10
    width: int = 1920
    height: int = 1080
    buffer_size: int = 1
    reconnect_delay_s: float = 2.0
    max_consecutive_failures: int = 30
    zone: str = "default"

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        mapped = dict(d)
        if "id" in mapped and "camera_id" not in mapped:
            mapped["camera_id"] = mapped.pop("id")
        return cls(**{k: v for k, v in mapped.items() if k in cls.__dataclass_fields__})
