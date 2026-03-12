"""
Detection data models for bounding boxes and multi-stage inference results.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundingBox:
    """Normalised [0,1] coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(x1=float(d["x1"]), y1=float(d["y1"]),
                   x2=float(d["x2"]), y2=float(d["y2"]))


@dataclass
class Detection:
    """
    A single detection from one stage of the inference pipeline.
    Stages: "primary" (e.g. vehicle/person) → "secondary" (e.g. plate/badge OCR)
    """
    detection_id: str
    frame_id: str
    camera_id: str
    stage: str                              # "primary" | "secondary"
    label: str                              # e.g. "vehicle", "person", "license_plate"
    confidence: float
    bbox: BoundingBox
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_detection_id: Optional[str] = None   # Links secondary → primary
    ocr_text: Optional[str] = None              # Result from OCR secondary stage
    track_id: Optional[str] = None              # Optional object tracking ID

    @classmethod
    def create(
        cls,
        frame_id: str,
        camera_id: str,
        stage: str,
        label: str,
        confidence: float,
        bbox: BoundingBox,
        parent_detection_id: Optional[str] = None,
        ocr_text: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Detection":
        return cls(
            detection_id=str(uuid.uuid4()),
            frame_id=frame_id,
            camera_id=camera_id,
            stage=stage,
            label=label,
            confidence=confidence,
            bbox=bbox,
            attributes=attributes or {},
            parent_detection_id=parent_detection_id,
            ocr_text=ocr_text,
        )

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "stage": self.stage,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "attributes": self.attributes,
            "parent_detection_id": self.parent_detection_id,
            "ocr_text": self.ocr_text,
            "track_id": self.track_id,
        }


@dataclass
class StageResult:
    """Aggregated output from a single inference stage for one frame."""
    frame_id: str
    camera_id: str
    stage: str
    detections: List[Detection]
    inference_latency_ms: float
    model_id: str

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "stage": self.stage,
            "detections": [d.to_dict() for d in self.detections],
            "inference_latency_ms": self.inference_latency_ms,
            "model_id": self.model_id,
        }
