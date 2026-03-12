"""
Inference Service — independent FastAPI microservice.

Exposes two endpoints:
  POST /detect/primary   → Stage A: object detection (vehicle, person, …)
  POST /detect/secondary → Stage B: crop-level classifier / OCR

In a real deployment these call YOLO / DETR / Tesseract / PaddleOCR.
This implementation is a STUB that returns realistic-looking random results
so the rest of the pipeline can be validated end-to-end.

Run with:
  uvicorn inference_service.app:app --host 0.0.0.0 --port 8001 --workers 2
"""
from __future__ import annotations

import base64
import io
import logging
import random
import time
import uuid
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="EAIGLE Inference Service", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class FramePayload(BaseModel):
    frame_id: str
    camera_id: str
    # Frame encoded as base64 numpy bytes (tobytes()) + shape/dtype metadata
    data_b64: str
    shape: List[int]          # [H, W, C]
    dtype: str                # "float32"
    stage: str = "primary"    # "primary" | "secondary"
    parent_detection_id: str | None = None
    label_hint: str | None = None   # e.g. "vehicle" to help secondary stage


class BatchRequest(BaseModel):
    batch_id: str
    frames: List[FramePayload]
    model_id: str = "default"


class BBoxResult(BaseModel):
    x1: float; y1: float; x2: float; y2: float


class DetectionResult(BaseModel):
    detection_id: str
    frame_id: str
    label: str
    confidence: float
    bbox: BBoxResult
    ocr_text: str | None = None
    attributes: Dict[str, Any] = {}
    parent_detection_id: str | None = None


class InferenceResponse(BaseModel):
    batch_id: str
    results: List[DetectionResult]
    inference_latency_ms: float
    model_id: str


# ---------------------------------------------------------------------------
# Stub detectors
# ---------------------------------------------------------------------------

_PRIMARY_LABELS = ["vehicle", "person", "truck", "bicycle", "motorcycle"]
_SECONDARY_LABELS = {
    "vehicle": "license_plate",
    "truck":   "license_plate",
    "person":  "badge",
}

# Simulate OCR outputs
_PLATE_SAMPLES = ["ABC123", "XYZ789", "TRK456", "CAM001", "PRC999"]
_BADGE_SAMPLES = ["ID-4821", "STAFF-002", "VIS-093", "MAINT-77", None]


def _stub_primary_detect(frame_id: str) -> List[DetectionResult]:
    """Return 0–3 random primary detections."""
    results = []
    num = random.randint(0, 3)
    for _ in range(num):
        label = random.choice(_PRIMARY_LABELS)
        x1, y1 = random.uniform(0, 0.5), random.uniform(0, 0.5)
        x2 = min(1.0, x1 + random.uniform(0.1, 0.4))
        y2 = min(1.0, y1 + random.uniform(0.1, 0.4))
        results.append(DetectionResult(
            detection_id=str(uuid.uuid4()),
            frame_id=frame_id,
            label=label,
            confidence=round(random.uniform(0.55, 0.99), 3),
            bbox=BBoxResult(x1=x1, y1=y1, x2=x2, y2=y2),
        ))
    return results


def _stub_secondary_detect(
    frame_id: str,
    label_hint: str | None,
    parent_detection_id: str | None,
) -> List[DetectionResult]:
    """Return 0–1 secondary detections (OCR / classifier)."""
    if random.random() < 0.2:   # 20% chance: nothing found in crop
        return []

    hint = label_hint or "vehicle"
    sec_label = _SECONDARY_LABELS.get(hint, "unknown_crop")

    if hint in ("vehicle", "truck"):
        ocr = random.choice(_PLATE_SAMPLES)
    elif hint == "person":
        ocr = random.choice(_BADGE_SAMPLES)
    else:
        ocr = None

    return [DetectionResult(
        detection_id=str(uuid.uuid4()),
        frame_id=frame_id,
        label=sec_label,
        confidence=round(random.uniform(0.6, 0.98), 3),
        bbox=BBoxResult(x1=0.05, y1=0.05, x2=0.95, y2=0.95),   # crop-relative
        ocr_text=ocr,
        parent_detection_id=parent_detection_id,
    )]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/detect/primary", response_model=InferenceResponse)
async def detect_primary(req: BatchRequest) -> InferenceResponse:
    t0 = time.monotonic()

    # Simulate GPU inference time (~5–30ms per batch)
    batch_size = len(req.frames)
    simulated_ms = random.uniform(5, 15) + batch_size * random.uniform(1, 3)
    import asyncio
    await asyncio.sleep(simulated_ms / 1000)

    all_results: List[DetectionResult] = []
    for frame in req.frames:
        all_results.extend(_stub_primary_detect(frame.frame_id))

    latency = (time.monotonic() - t0) * 1000
    logger.info(
        "Primary batch %s: %d frames → %d detections in %.1fms",
        req.batch_id, batch_size, len(all_results), latency,
    )
    return InferenceResponse(
        batch_id=req.batch_id,
        results=all_results,
        inference_latency_ms=latency,
        model_id=req.model_id,
    )


@app.post("/detect/secondary", response_model=InferenceResponse)
async def detect_secondary(req: BatchRequest) -> InferenceResponse:
    t0 = time.monotonic()

    batch_size = len(req.frames)
    simulated_ms = random.uniform(3, 10) + batch_size * random.uniform(0.5, 2)
    import asyncio
    await asyncio.sleep(simulated_ms / 1000)

    all_results: List[DetectionResult] = []
    for frame in req.frames:
        all_results.extend(
            _stub_secondary_detect(
                frame.frame_id,
                frame.label_hint,
                frame.parent_detection_id,
            )
        )

    latency = (time.monotonic() - t0) * 1000
    logger.info(
        "Secondary batch %s: %d crops → %d results in %.1fms",
        req.batch_id, batch_size, len(all_results), latency,
    )
    return InferenceResponse(
        batch_id=req.batch_id,
        results=all_results,
        inference_latency_ms=latency,
        model_id=req.model_id,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "inference"}
