"""
Spatial Non-Maximum Suppression (NMS).

Removes overlapping bounding boxes within the same detection stage,
keeping the highest-confidence box when IoU exceeds the threshold.
"""
from __future__ import annotations

from typing import List

from eaigle.models.detection import Detection


class SpatialNMS:
    def __init__(self, iou_threshold: float = 0.45):
        self._iou_threshold = iou_threshold

    def apply(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            sorted_dets = [
                d for d in sorted_dets
                if best.bbox.iou(d.bbox) < self._iou_threshold
            ]

        return kept
