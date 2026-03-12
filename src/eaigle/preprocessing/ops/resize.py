from __future__ import annotations

import cv2
import numpy as np

from eaigle.preprocessing.ops.base_op import BaseOp

_INTERP_MAP = {
    "LINEAR": cv2.INTER_LINEAR,
    "NEAREST": cv2.INTER_NEAREST,
    "CUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS": cv2.INTER_LANCZOS4,
}


class ResizeOp(BaseOp):
    """Resize frame to (width, height) using configurable interpolation."""

    def __init__(self, width: int, height: int, interpolation: str = "LINEAR"):
        self._size = (width, height)
        self._interp = _INTERP_MAP.get(interpolation.upper(), cv2.INTER_LINEAR)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self._size, interpolation=self._interp)
