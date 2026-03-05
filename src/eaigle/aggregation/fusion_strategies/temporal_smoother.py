
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Deque, List

from eaigle.models.detection import Detection

class TemporalSmoother:
    def __init__(self, window: int = 5, min_presence: int = 2):
        self._window = window
        self._min_presence = min_presence

        self._history: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window))
        )

    def smooth(
        self, camera_id: str, detections: List[Detection]
    ) -> List[Detection]:
        cam_history = self._history[camera_id]
        seen_labels = {d.label for d in detections}

        for label in seen_labels:
            det = next(d for d in detections if d.label == label)
            cam_history[label].append(det.confidence)

        for label in list(cam_history.keys()):
            if label not in seen_labels:
                cam_history[label].append(0.0)

        confirmed: List[Detection] = []
        for det in detections:
            history = cam_history[det.label]
            appearances = sum(1 for c in history if c > 0)
            if appearances >= self._min_presence:
                confirmed.append(det)

        return confirmed
