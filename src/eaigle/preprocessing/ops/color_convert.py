from __future__ import annotations

import cv2
import numpy as np

from eaigle.preprocessing.ops.base_op import BaseOp

_CONVERT_MAP = {
    ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
    ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
    ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
    ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
    ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
}


class ColorConvertOp(BaseOp):
    """Convert between colour spaces (e.g. BGR→RGB for most DL models)."""

    def __init__(self, src: str = "BGR", dst: str = "RGB"):
        key = (src.upper(), dst.upper())
        if key not in _CONVERT_MAP:
            raise ValueError(f"Unsupported colour conversion: {src} → {dst}")
        self._code = _CONVERT_MAP[key]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, self._code)
