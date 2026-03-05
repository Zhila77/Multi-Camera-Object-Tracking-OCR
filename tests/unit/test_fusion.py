"""Unit tests for aggregation fusion strategies and HypothesisFusion."""
import time

import pytest

from eaigle.aggregation.fusion_strategies.confidence_voting import ConfidenceVoting
from eaigle.aggregation.fusion_strategies.spatial_nms import SpatialNMS
from eaigle.aggregation.fusion_strategies.temporal_smoother import TemporalSmoother
from eaigle.aggregation.hypothesis_fusion import HypothesisFusion
from eaigle.models.detection import BoundingBox, Detection, StageResult


def _det(label="vehicle", confidence=0.9, x1=0.1, y1=0.1, x2=0.5, y2=0.5,
         stage="primary", fid="f1", cid="cam_01"):
    return Detection.create(
        frame_id=fid, camera_id=cid, stage=stage,
        label=label, confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


class TestSpatialNMS:
    def test_removes_high_iou_duplicate(self):
        dets = [
            _det(confidence=0.9, x1=0.1, y1=0.1, x2=0.5, y2=0.5),
            _det(confidence=0.7, x1=0.12, y1=0.12, x2=0.52, y2=0.52),  # ~0.88 IoU
        ]
        nms = SpatialNMS(iou_threshold=0.5)
        result = nms.apply(dets)
        assert len(result) == 1
        assert result[0].confidence == 0.9   # higher confidence kept

    def test_keeps_non_overlapping(self):
        dets = [
            _det(x1=0.0, y1=0.0, x2=0.3, y2=0.3, confidence=0.9),
            _det(x1=0.7, y1=0.7, x2=1.0, y2=1.0, confidence=0.8),
        ]
        nms = SpatialNMS(iou_threshold=0.5)
        result = nms.apply(dets)
        assert len(result) == 2

    def test_empty_input(self):
        assert SpatialNMS().apply([]) == []


class TestConfidenceVoting:
    def test_filters_below_threshold(self):
        dets = [_det(confidence=0.8), _det(confidence=0.3), _det(confidence=0.6)]
        cv = ConfidenceVoting(min_confidence=0.5)
        result = cv.apply(dets)
        assert all(d.confidence >= 0.5 for d in result)
        assert len(result) == 2

    def test_sorted_descending(self):
        dets = [_det(confidence=0.6), _det(confidence=0.9), _det(confidence=0.75)]
        result = ConfidenceVoting(min_confidence=0.0).apply(dets)
        confs = [d.confidence for d in result]
        assert confs == sorted(confs, reverse=True)


class TestTemporalSmoother:
    def test_single_appearance_suppressed(self):
        smoother = TemporalSmoother(window=5, min_presence=2)
        dets = [_det(label="vehicle", confidence=0.9)]
        # First frame: appears once
        result = smoother.smooth("cam_01", dets)
        assert result == []   # Not yet confirmed

    def test_confirmed_after_min_presence(self):
        smoother = TemporalSmoother(window=5, min_presence=2)
        det = _det(label="vehicle", confidence=0.9)
        smoother.smooth("cam_01", [det])  # frame 1
        result = smoother.smooth("cam_01", [det])  # frame 2
        assert len(result) == 1   # Now confirmed


class TestHypothesisFusion:
    def _make_stage_result(self, stage, dets):
        return StageResult(
            frame_id="f1", camera_id="cam_01",
            stage=stage, detections=dets,
            inference_latency_ms=10.0, model_id="stub",
        )

    def test_produces_hypothesis(self):
        fusion = HypothesisFusion(
            temporal_min_presence=1,   # confirm on first appearance
            camera_zones={"cam_01": "Gate 1"},
        )
        primary_dets = [_det(label="vehicle", confidence=0.85)]
        stage_results = [
            self._make_stage_result("primary", primary_dets),
        ]
        hyp = fusion.fuse("f1", "cam_01", time.time(), stage_results)
        assert hyp.frame_id == "f1"
        assert hyp.camera_id == "cam_01"
        assert "Gate 1" in hyp.event_description
        assert hyp.overall_confidence > 0

    def test_empty_detections_no_crash(self):
        fusion = HypothesisFusion(temporal_min_presence=1)
        hyp = fusion.fuse("f2", "cam_02", time.time(), [])
        assert "No objects" in hyp.event_description
        assert hyp.overall_confidence == 0.0
