
from __future__ import annotations

import base64
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="EAIGLE Inference Service", version="0.2.0")

_YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
_yolo = None

def _get_yolo():
    global _yolo
    if _yolo is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO model: %s", _YOLO_MODEL_NAME)
            _yolo = YOLO(_YOLO_MODEL_NAME)
            logger.info("YOLO model loaded")
        except ImportError:
            logger.warning("ultralytics not installed — falling back to stub")
    return _yolo

@app.on_event("startup")
async def _startup():
    _get_yolo()

_COCO_TO_LABEL: Dict[str, str] = {
    "person":       "person",
    "bicycle":      "bicycle",
    "car":          "vehicle",
    "motorcycle":   "motorcycle",
    "bus":          "vehicle",
    "truck":        "truck",
    "traffic light": "traffic_light",
    "stop sign":    "stop_sign",
}

_KEEP_CLASSES = set(_COCO_TO_LABEL.keys())

class FramePayload(BaseModel):
    frame_id: str
    camera_id: str
    data_b64: str
    shape: List[int]
    dtype: str
    stage: str = "primary"
    parent_detection_id: str | None = None
    label_hint: str | None = None

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

_DTYPE_ALIASES: Dict[str, str] = {
    "string": "uint8",
    "str": "uint8",
    "bytes": "uint8",
    "object": "uint8",
}

def _resolve_dtype(dtype_str: str) -> str:
    
    try:
        np.dtype(dtype_str)
        return dtype_str
    except TypeError:
        resolved = _DTYPE_ALIASES.get(dtype_str, "uint8")
        logger.warning("Unknown dtype %r — treating as %s", dtype_str, resolved)
        return resolved

def _decode_frame(payload: FramePayload) -> np.ndarray:
    
    data = payload.data_b64

    data += "=" * (-len(data) % 4)
    raw = base64.b64decode(data)
    dtype = _resolve_dtype(payload.dtype)
    shape = payload.shape
    if not shape or any(d == 0 for d in shape):
        logger.warning("Skipping frame %s — invalid shape %s", payload.frame_id, shape)
        return np.zeros((1, 1, 3), dtype=np.uint8)
    try:
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except ValueError as exc:
        logger.warning(
            "Cannot reshape frame %s: %d bytes → shape %s (%s). Returning blank.",
            payload.frame_id, len(raw), shape, exc,
        )
        return np.zeros((shape[0] or 1, shape[1] if len(shape) > 1 else 1, 3), dtype=np.uint8)

    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        span = hi - lo
        if span > 0:
            arr = ((arr - lo) / span * 255).astype(np.uint8)
        else:
            arr = np.zeros(arr.shape, dtype=np.uint8)

    return arr

def _yolo_detect(frame_id: str, img: np.ndarray, yolo) -> List[DetectionResult]:
    

    results = yolo.predict(img, verbose=False, conf=0.35)

    detections: List[DetectionResult] = []
    h, w = img.shape[:2]

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_name = result.names[int(box.cls[0])]
            if cls_name not in _KEEP_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(DetectionResult(
                detection_id=str(uuid.uuid4()),
                frame_id=frame_id,
                label=_COCO_TO_LABEL[cls_name],
                confidence=round(float(box.conf[0]), 3),
                bbox=BBoxResult(
                    x1=x1 / w,
                    y1=y1 / h,
                    x2=x2 / w,
                    y2=y2 / h,
                ),
            ))

    return detections

_SECONDARY_LABELS = {"vehicle": "license_plate", "truck": "license_plate", "person": "badge"}
_PLATE_SAMPLES = ["ABC123", "XYZ789", "TRK456", "CAM001", "PRC999"]
_BADGE_SAMPLES = ["ID-4821", "STAFF-002", "VIS-093", "MAINT-77", None]

def _stub_secondary_detect(
    frame_id: str,
    label_hint: str | None,
    parent_detection_id: str | None,
) -> List[DetectionResult]:
    if random.random() < 0.2:
        return []
    hint = label_hint or "vehicle"
    sec_label = _SECONDARY_LABELS.get(hint, "unknown_crop")
    ocr = random.choice(_PLATE_SAMPLES if hint in ("vehicle", "truck") else _BADGE_SAMPLES)
    return [DetectionResult(
        detection_id=str(uuid.uuid4()),
        frame_id=frame_id,
        label=sec_label,
        confidence=round(random.uniform(0.6, 0.98), 3),
        bbox=BBoxResult(x1=0.05, y1=0.05, x2=0.95, y2=0.95),
        ocr_text=ocr,
        parent_detection_id=parent_detection_id,
    )]

@app.post("/detect/primary", response_model=InferenceResponse)
async def detect_primary(req: BatchRequest) -> InferenceResponse:
    t0 = time.monotonic()
    yolo = _get_yolo()
    all_results: List[DetectionResult] = []

    for frame in req.frames:
        img = _decode_frame(frame)
        if yolo is not None:
            all_results.extend(_yolo_detect(frame.frame_id, img, yolo))
        else:

            num = random.randint(0, 2)
            for _ in range(num):
                label = random.choice(["vehicle", "person", "truck"])
                x1, y1 = random.uniform(0, 0.5), random.uniform(0, 0.5)
                all_results.append(DetectionResult(
                    detection_id=str(uuid.uuid4()),
                    frame_id=frame.frame_id,
                    label=label,
                    confidence=round(random.uniform(0.55, 0.90), 3),
                    bbox=BBoxResult(x1=x1, y1=y1,
                                    x2=min(1.0, x1 + 0.3),
                                    y2=min(1.0, y1 + 0.3)),
                ))

    latency = (time.monotonic() - t0) * 1000
    logger.info(
        "Primary batch %s: %d frames → %d detections in %.1fms",
        req.batch_id, len(req.frames), len(all_results), latency,
    )
    return InferenceResponse(
        batch_id=req.batch_id,
        results=all_results,
        inference_latency_ms=latency,
        model_id=f"yolo:{_YOLO_MODEL_NAME}" if yolo else "stub",
    )

@app.post("/detect/secondary", response_model=InferenceResponse)
async def detect_secondary(req: BatchRequest) -> InferenceResponse:
    t0 = time.monotonic()
    all_results: List[DetectionResult] = []
    for frame in req.frames:
        all_results.extend(
            _stub_secondary_detect(frame.frame_id, frame.label_hint, frame.parent_detection_id)
        )
    latency = (time.monotonic() - t0) * 1000
    logger.info(
        "Secondary batch %s: %d crops → %d results in %.1fms",
        req.batch_id, len(req.frames), len(all_results), latency,
    )
    return InferenceResponse(
        batch_id=req.batch_id,
        results=all_results,
        inference_latency_ms=latency,
        model_id="stub:ocr",
    )

@app.get("/health")
async def health() -> dict:
    yolo = _get_yolo()
    return {
        "status": "ok",
        "service": "inference",
        "model": _YOLO_MODEL_NAME if yolo else "stub",
    }
