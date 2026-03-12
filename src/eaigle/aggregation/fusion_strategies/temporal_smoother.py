"""
Temporal Smoother — stabilises detections over time per camera.

Uses a sliding window of the last N frames.  A label that appeared in
at least *min_presence* of those frames is confirmed; sporadic detections
(single-frame flashes) are suppressed.

This prevents flickering hypotheses from noisy models.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Deque, List

from eaigle.models.detection import Detection


class TemporalSmoother:
    def __init__(self, window: int = 5, min_presence: int = 2):
        self._window = window
        self._min_presence = min_presence
        # camera_id → label → deque of (confidence, appeared)
        self._history: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window))
        )

    def smooth(
        self, camera_id: str, detections: List[Detection]
    ) -> List[Detection]:
        cam_history = self._history[camera_id]
        seen_labels = {d.label for d in detections}

        # Record current frame's detections
        for label in seen_labels:
            det = next(d for d in detections if d.label == label)
            cam_history[label].append(det.confidence)

        # Labels not seen this frame get a zero entry
        for label in list(cam_history.keys()):
            if label not in seen_labels:
                cam_history[label].append(0.0)

        # Keep detections whose label has appeared ≥ min_presence times
        confirmed: List[Detection] = []
        for det in detections:
            history = cam_history[det.label]
            appearances = sum(1 for c in history if c > 0)
            if appearances >= self._min_presence:
                confirmed.append(det)

        return confirmed
