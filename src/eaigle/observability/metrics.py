"""
Prometheus metrics for the pipeline.

All metrics are created once at module import time and shared across
the process.  The HTTP server is started by calling start_metrics_server().

Key metrics:
  - eaigle_frames_captured_total        — ingestion throughput per camera
  - eaigle_frames_preprocessed_total    — preprocessing throughput
  - eaigle_inference_batch_size         — batch efficiency histogram
  - eaigle_inference_latency_ms         — per-stage latency histogram
  - eaigle_pipeline_latency_ms          — end-to-end frame latency
  - eaigle_camera_reconnects_total      — reliability indicator
  - eaigle_dropped_frames_total         — data loss counter
  - eaigle_redis_stream_lag             — consumer group lag gauge
  - eaigle_hypotheses_total             — throughput of final outputs
"""
from __future__ import annotations

import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ---- Counters ----
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
    ["reason"],  # "shm_write", "publish_error", "backpressure"
)

HYPOTHESES_PRODUCED = Counter(
    "eaigle_hypotheses_total",
    "Total hypotheses produced by the aggregation layer",
    ["camera_id"],
)

# ---- Histograms ----
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

# ---- Gauges ----
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
    """Start the Prometheus metrics HTTP server in a background daemon thread."""
    start_http_server(port)
