from __future__ import annotations

import numpy as np

from eaigle.preprocessing.ops.base_op import BaseOp


class NormalizeOp(BaseOp):
    """
    Normalise pixel values to [0,1] then apply per-channel mean/std subtraction
    (ImageNet defaults).  Produces float32 output suitable for DL models.
    """

    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        scale: float = 1.0 / 255.0,
    ):
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)
        self._scale = scale

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32) * self._scale
        frame = (frame - self._mean) / self._std
        return frame
