"""Unit tests for core data models."""
import time

import pytest

from eaigle.models.detection import BoundingBox, Detection
from eaigle.models.frame import FrameMetadata
from eaigle.models.hypothesis import CameraConfig, Hypothesis


class TestBoundingBox:
    def test_area(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.4)
        assert abs(bbox.area - 0.2) < 1e-9

    def test_iou_identical_boxes(self):
        bbox = BoundingBox(0.1, 0.1, 0.5, 0.5)
        assert bbox.iou(bbox) == pytest.approx(1.0)

    def test_iou_non_overlapping(self):
        a = BoundingBox(0.0, 0.0, 0.3, 0.3)
        b = BoundingBox(0.7, 0.7, 1.0, 1.0)
        assert a.iou(b) == 0.0

    def test_to_pixel_coords(self):
        bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
        assert bbox.to_pixel_coords(640, 480) == (0, 0, 320, 240)


class TestFrameMetadata:
    def test_round_trip_dict(self):
        meta = FrameMetadata.create(
            camera_id="cam_01",
            sequence_num=42,
            shm_key="eaigle_xyz",
            shape=(480, 640, 3),
        )
        recovered = FrameMetadata.from_dict(meta.to_dict())
        assert recovered.frame_id == meta.frame_id
        assert recovered.camera_id == "cam_01"
        assert recovered.sequence_num == 42
        assert recovered.width == 640
        assert recovered.height == 480


class TestCameraConfig:
    def test_from_dict(self):
        d = {
            "camera_id": "cam_01",
            "rtsp_url": "rtsp://localhost/stream",
            "target_fps": 15,
            "zone": "Gate 3",
        }
        cfg = CameraConfig.from_dict(d)
        assert cfg.camera_id == "cam_01"
        assert cfg.target_fps == 15
        assert cfg.zone == "Gate 3"
