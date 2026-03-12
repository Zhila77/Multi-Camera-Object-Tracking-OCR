"""
Confidence-weighted filtering.

Removes detections below the minimum threshold and returns
the remainder sorted by confidence descending.
"""
from __future__ import annotations

from typing import List

from eaigle.models.detection import Detection


class ConfidenceVoting:
    def __init__(self, min_confidence: float = 0.40):
        self._min_conf = min_confidence

    def apply(self, detections: List[Detection]) -> List[Detection]:
        filtered = [d for d in detections if d.confidence >= self._min_conf]
        return sorted(filtered, key=lambda d: d.confidence, reverse=True)
