
from __future__ import annotations

import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

FRAMES_CAPTURED = Counter(
    "eaigle_frames_captured_total",
    "Total frames captured from RTSP cameras",
    ["camera_id"],
)

FRAMES_PREPROCESSED = Counter(
    "eaigle_frames_preprocessed_total",
    "Total frames successfully preprocessed",
    ["camera_id"],
)

CAMERA_RECONNECTS = Counter(
    "eaigle_camera_reconnects_total",
    "Number of camera reconnect attempts",
    ["camera_id"],
)

DROPPED_FRAMES = Counter(
    "eaigle_dropped_frames_total",
    "Frames dropped due to errors or back-pressure",
    ["reason"],
)

HYPOTHESES_PRODUCED = Counter(
    "eaigle_hypotheses_total",
    "Total hypotheses produced by the aggregation layer",
    ["camera_id"],
)

PREPROCESS_LATENCY = Histogram(
    "eaigle_preprocess_latency_ms",
    "Time spent in preprocessing per frame (ms)",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
)

INFERENCE_LATENCY = Histogram(
    "eaigle_inference_latency_ms",
    "Inference service round-trip latency (ms)",
    ["stage"],
    buckets=[5, 10, 20, 50, 100, 200, 500],
)

INFERENCE_BATCH_SIZE = Histogram(
    "eaigle_inference_batch_size",
    "Number of frames per inference batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

PIPELINE_LATENCY = Histogram(
    "eaigle_pipeline_latency_ms",
    "End-to-end latency from frame capture to hypothesis (ms)",
    ["camera_id"],
    buckets=[50, 100, 200, 300, 500, 1000, 2000],
)

REDIS_STREAM_LAG = Gauge(
    "eaigle_redis_stream_lag",
    "Pending message count in Redis Stream consumer group",
    ["stream"],
)

ACTIVE_CAMERAS = Gauge(
    "eaigle_active_cameras",
    "Number of camera streams currently connected",
)

def start_metrics_server(port: int = 9090) -> None:
    
    start_http_server(port)
