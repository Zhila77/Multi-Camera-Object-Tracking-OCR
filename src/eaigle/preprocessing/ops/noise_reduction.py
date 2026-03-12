from __future__ import annotations

import cv2
import numpy as np

from eaigle.preprocessing.ops.base_op import BaseOp


class GaussianDenoiseOp(BaseOp):
    """
    Gaussian blur for mild noise reduction.
    Cheap CPU op; suitable for real-time pipelines.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.0):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self._ksize = (kernel_size, kernel_size)
        self._sigma = sigma

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, self._ksize, self._sigma)


class FastNLMeansDenoiseOp(BaseOp):
    """
    Non-local means denoising (higher quality, higher cost).
    Suitable when GPU is unavailable and noise is severe.
    """

    def __init__(self, h: float = 10.0, template_window: int = 7, search_window: int = 21):
        self._h = h
        self._tws = template_window
        self._sws = search_window

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            frame, None, self._h, self._h, self._tws, self._sws
        )
