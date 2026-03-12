"""
HypothesisFusion — aggregates all StageResults for a single frame into
one Hypothesis with a human-readable event description.

Fusion pipeline:
  1. SpatialNMS        — remove overlapping boxes within each stage
  2. ConfidenceVoting  — filter below minimum confidence
  3. TemporalSmoother  — suppress flickering (requires history across frames)
  4. Event narration   — produce a structured event description string
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from eaigle.aggregation.fusion_strategies.confidence_voting import ConfidenceVoting
from eaigle.aggregation.fusion_strategies.spatial_nms import SpatialNMS
from eaigle.aggregation.fusion_strategies.temporal_smoother import TemporalSmoother
from eaigle.models.detection import Detection, StageResult
from eaigle.models.hypothesis import CameraConfig, Hypothesis

logger = logging.getLogger(__name__)


class HypothesisFusion:
    def __init__(
        self,
        nms_iou_threshold: float = 0.45,
        min_confidence: float = 0.40,
        temporal_window: int = 5,
        temporal_min_presence: int = 2,
        camera_zones: Optional[Dict[str, str]] = None,
    ):
        self._nms = SpatialNMS(iou_threshold=nms_iou_threshold)
        self._voting = ConfidenceVoting(min_confidence=min_confidence)
        self._smoother = TemporalSmoother(
            window=temporal_window, min_presence=temporal_min_presence
        )
        self._zones: Dict[str, str] = camera_zones or {}

    def fuse(
        self,
        frame_id: str,
        camera_id: str,
        capture_ts: float,
        stage_results: List[StageResult],
    ) -> Hypothesis:
        # Split results by stage
        by_stage: Dict[str, List[Detection]] = {}
        for sr in stage_results:
            by_stage.setdefault(sr.stage, []).extend(sr.detections)

        # --- Stage A (primary) fusion ---
        primary = by_stage.get("primary", [])
        primary = self._nms.apply(primary)
        primary = self._voting.apply(primary)
        primary = self._smoother.smooth(camera_id, primary)

        # --- Stage B (secondary / OCR) fusion ---
        secondary = by_stage.get("secondary", [])
        secondary = self._voting.apply(secondary)   # NMS not needed for crops

        # --- Overall confidence ---
        all_dets = primary + secondary
        overall_conf = (
            sum(d.confidence for d in all_dets) / len(all_dets)
            if all_dets else 0.0
        )

        # --- Event narration ---
        zone = self._zones.get(camera_id, "unknown zone")
        event_desc = self._narrate(camera_id, zone, primary, secondary)

        logger.info(
            "Hypothesis for frame %s @ %s: %d primary, %d secondary — %s",
            frame_id, camera_id, len(primary), len(secondary), event_desc,
        )

        return Hypothesis.create(
            frame_id=frame_id,
            camera_id=camera_id,
            capture_ts=capture_ts,
            primary_detections=primary,
            secondary_detections=secondary,
            event_description=event_desc,
            overall_confidence=round(overall_conf, 4),
            fusion_strategy="nms+confidence+temporal",
        )

    # ------------------------------------------------------------------
    # Event narration
    # ------------------------------------------------------------------

    def _narrate(
        self,
        camera_id: str,
        zone: str,
        primary: List[Detection],
        secondary: List[Detection],
    ) -> str:
        if not primary:
            return f"No objects detected at {zone} ({camera_id})"

        parts: List[str] = []
        ocr_by_parent = {
            d.parent_detection_id: d for d in secondary if d.ocr_text
        }

        for det in primary:
            label = det.label.capitalize()
            conf_pct = int(det.confidence * 100)
            ocr_det = ocr_by_parent.get(det.detection_id)

            if ocr_det and ocr_det.ocr_text:
                ocr_label = ocr_det.label.replace("_", " ")
                parts.append(
                    f"{label} '{ocr_det.ocr_text}' ({ocr_label}, {conf_pct}%)"
                )
            else:
                parts.append(f"{label} ({conf_pct}%)")

        subjects = "; ".join(parts)
        return f"Detected at {zone} [{camera_id}]: {subjects}"
