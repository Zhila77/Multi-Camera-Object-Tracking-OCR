"""Unit tests for preprocessing operations and pipeline."""
import numpy as np
import pytest

from eaigle.preprocessing.ops.color_convert import ColorConvertOp
from eaigle.preprocessing.ops.noise_reduction import GaussianDenoiseOp
from eaigle.preprocessing.ops.normalize import NormalizeOp
from eaigle.preprocessing.ops.resize import ResizeOp
from eaigle.preprocessing.pipeline import PreprocessingPipeline


@pytest.fixture
def bgr_frame():
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


class TestResizeOp:
    def test_output_shape(self, bgr_frame):
        op = ResizeOp(width=320, height=240)
        out = op(bgr_frame)
        assert out.shape == (240, 320, 3)

    def test_preserves_channels(self, bgr_frame):
        op = ResizeOp(width=100, height=100)
        out = op(bgr_frame)
        assert out.shape[2] == 3


class TestColorConvertOp:
    def test_bgr_to_rgb_swaps_channels(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 100   # B
        frame[:, :, 2] = 200   # R
        op = ColorConvertOp(src="BGR", dst="RGB")
        out = op(frame)
        assert out[0, 0, 0] == 200   # R now at index 0
        assert out[0, 0, 2] == 100   # B now at index 2

    def test_invalid_conversion_raises(self):
        with pytest.raises(ValueError):
            ColorConvertOp(src="XYZ", dst="ABC")


class TestNormalizeOp:
    def test_output_dtype_float32(self, bgr_frame):
        op = NormalizeOp()
        out = op(bgr_frame)
        assert out.dtype == np.float32

    def test_zero_pixel_normalised(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        op = NormalizeOp(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), scale=1.0 / 255.0)
        out = op(frame)
        assert np.allclose(out, 0.0)


class TestGaussianDenoiseOp:
    def test_output_shape_preserved(self, bgr_frame):
        op = GaussianDenoiseOp(kernel_size=3)
        out = op(bgr_frame)
        assert out.shape == bgr_frame.shape

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            GaussianDenoiseOp(kernel_size=4)


class TestPreprocessingPipeline:
    def test_chain_runs_in_order(self, bgr_frame):
        pipeline = PreprocessingPipeline.from_config({
            "ops": [
                {"type": "color_convert", "params": {"src": "BGR", "dst": "RGB"}},
                {"type": "resize",        "params": {"width": 320, "height": 240}},
                {"type": "normalize",     "params": {}},
            ]
        })
        out = pipeline.run(bgr_frame)
        assert out.shape == (240, 320, 3)
        assert out.dtype == np.float32

    def test_empty_pipeline_is_identity(self, bgr_frame):
        pipeline = PreprocessingPipeline.from_config({"ops": []})
        out = pipeline.run(bgr_frame)
        assert np.array_equal(out, bgr_frame)
