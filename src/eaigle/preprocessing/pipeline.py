"""
Composable preprocessing pipeline.

Each op is a callable: np.ndarray → np.ndarray.
The pipeline is stateless and safe to use across processes.

Example config:
  ops:
    - type: color_convert
      params: {src: BGR, dst: RGB}
    - type: resize
      params: {width: 640, height: 640}
    - type: normalize
      params: {}
"""
from __future__ import annotations

from typing import List

import numpy as np

from eaigle.preprocessing.ops.base_op import BaseOp


def _build_op(op_cfg: dict) -> BaseOp:
    op_type = op_cfg["type"]
    params = op_cfg.get("params", {})

    if op_type == "color_convert":
        from eaigle.preprocessing.ops.color_convert import ColorConvertOp
        return ColorConvertOp(**params)
    if op_type == "resize":
        from eaigle.preprocessing.ops.resize import ResizeOp
        return ResizeOp(**params)
    if op_type == "normalize":
        from eaigle.preprocessing.ops.normalize import NormalizeOp
        return NormalizeOp(**params)
    if op_type == "gaussian_denoise":
        from eaigle.preprocessing.ops.noise_reduction import GaussianDenoiseOp
        return GaussianDenoiseOp(**params)
    if op_type == "nlm_denoise":
        from eaigle.preprocessing.ops.noise_reduction import FastNLMeansDenoiseOp
        return FastNLMeansDenoiseOp(**params)

    raise ValueError(f"Unknown preprocessing op: {op_type}")


class PreprocessingPipeline:
    """Chain of preprocessing operations applied sequentially to each frame."""

    def __init__(self, ops: List[BaseOp]):
        self._ops = ops

    def run(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        for op in self._ops:
            result = op(result)
        return result

    @classmethod
    def from_config(cls, cfg: dict) -> "PreprocessingPipeline":
        ops = [_build_op(op_cfg) for op_cfg in cfg.get("ops", [])]
        return cls(ops)

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(ops={self._ops})"
