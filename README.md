# EAIGLE AI — Multi-Camera Computer Vision Pipeline
## Technical Report

**Project:** Real-Time Multi-Camera CV Pipeline
**Date:** 2026-03-04
**Stack:** Python 3.12, asyncio, Redis Streams, YOLOv8, FastAPI, Docker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component-Level Breakdown](#3-component-level-breakdown)
4. [Data Flow Description](#4-data-flow-description)
5. [Deployment Strategy](#5-deployment-strategy)
6. [Configuration Reference](#6-configuration-reference)
7. [How to Run](#7-how-to-run)
12. [Future Work](#12-future-work)

---

## 1. Project Overview

EAIGLE AI is a scalable real-time computer vision pipeline designed to ingest video streams from up to **50 simultaneous RTSP cameras**, run object detection and OCR, and produce structured event hypotheses for downstream consumers (dashboards, alert systems, audit logs).

### Key Requirements Addressed

| Requirement | Solution |
|---|---|
| 50 simultaneous RTSP cameras | asyncio + ThreadPoolExecutor per camera |
| Real-time preprocessing | ProcessPoolExecutor (GIL bypass, 4–8 CPU cores) |
| Zero-copy frame transfer | POSIX shared memory (`/dev/shm`) |
| At-least-once delivery | Redis Streams with consumer groups + XACK |
| Object detection | YOLOv8 (configurable: nano → xlarge) |
| Multi-stage cascade | Stage A (full frame) → Stage B (crops: OCR/classify) |
| Hypothesis deduplication | SpatialNMS + ConfidenceVoting + TemporalSmoother |
| GPU hardware decode (optional) | NVIDIA DeepStream backend |
| Observability | Prometheus metrics + structured JSON logs |
| Containerized deployment | Docker Compose (dev) / Kubernetes (prod) |

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EAIGLE AI PIPELINE                                   │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         ┌──────────┐            │
│  │  CAM_01  │  │  CAM_02  │  │  CAM_03  │   ...   │  CAM_50  │            │
│  │  RTSP    │  │  RTSP    │  │  RTSP    │         │  RTSP    │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         └────┬─────┘            │
│       └──────────────┴──────────────┴────────────────────┘                 │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │   INGESTION LAYER   │                                  │
│                    │  ┌───────────────┐ │                                  │
│                    │  │OpenCV+FFmpeg  │ │  ← asyncio + ThreadPoolExecutor  │
│                    │  │  OR DeepStream│ │  ← NVDEC hardware decode (GPU)   │
│                    │  └───────────────┘ │                                  │
│                    └─────────┬──────────┘                                  │
│                              │  frame pixels (6 MB per frame)              │
│              ┌───────────────┼────────────────┐                            │
│              ▼               ▼                ▼                            │
│       /dev/shm          /dev/shm          /dev/shm   ← POSIX Shared Mem   │
│    eaigle_frame_1    eaigle_frame_2    eaigle_frame_3                      │
│              └───────────────┴────────────────┘                            │
│                              │  metadata pointer only (~200 bytes)         │
│                    ┌─────────▼──────────┐                                  │
│                    │   Redis Streams     │  raw_frames:{cam_id}            │
│                    └─────────┬──────────┘                                  │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │ PREPROCESSING LAYER │                                  │
│                    │  ProcessPoolExecutor│  ← 4–8 workers (GIL bypass)     │
│                    │  color_convert      │                                  │
│                    │  resize → 640×640   │                                  │
│                    │  normalize float32  │                                  │
│                    └─────────┬──────────┘                                  │
│                              │  metadata pointer (new shm block)           │
│                    ┌─────────▼──────────┐                                  │
│                    │   Redis Streams     │  preprocessed_frames             │
│                    └─────────┬──────────┘                                  │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │ INFERENCE DISPATCH  │                                  │
│                    │  DynamicBatcher     │  ← batch ≤8 OR ≤50ms timeout    │
│                    └─────────┬──────────┘                                  │
│                              │  HTTP batch request (base64 frames)         │
│              ┌───────────────┴────────────────┐                            │
│              ▼                                ▼                            │
│   ┌──────────────────┐             ┌──────────────────┐                   │
│   │ /detect/primary  │             │ /detect/secondary│                   │
│   │  YOLOv8 Object   │──crops──►   │  OCR / Classify  │                   │
│   │  Detection       │             │  (license plates,│                   │
│   │  (vehicle,person)│             │   badges, etc.)  │                   │
│   └────────┬─────────┘             └────────┬─────────┘                   │
│            │    FastAPI Microservice (port 8001)    │                      │
│            └───────────────┬────────────────┘                             │
│                            │  StageResult JSON                             │
│                  ┌─────────▼──────────┐                                   │
│                  │   Redis Streams     │  inference_results                │
│                  └─────────┬──────────┘                                   │
│                            │                                               │
│                  ┌─────────▼──────────┐                                   │
│                  │ AGGREGATION LAYER   │                                   │
│                  │  SpatialNMS         │  ← IoU > 0.45 suppression        │
│                  │  ConfidenceVoting   │  ← score < 0.40 filter           │
│                  │  TemporalSmoother   │  ← 5-frame sliding window        │
│                  └─────────┬──────────┘                                   │
│                            │  Hypothesis events                            │
│                  ┌─────────▼──────────┐                                   │
│                  │   Redis PubSub      │  hypotheses channel               │
│                  └─────────┬──────────┘                                   │
│           ┌────────────────┼──────────────────┐                           │
│           ▼                ▼                  ▼                           │
│     Dashboard          Alert System       Audit Log                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component-Level Breakdown

### 3.1 Ingestion Layer

| Component | File | Role |
|---|---|---|
| `CameraWorker` | `src/eaigle/ingestion/camera_worker.py` | One asyncio coroutine per camera. Reads RTSP via `cv2.VideoCapture` offloaded to `ThreadPoolExecutor`. Auto-reconnects on failure. |
| `CameraStreamManager` | `src/eaigle/ingestion/camera_manager.py` | Supervises all N camera coroutines. Restarts crashed tasks every 2s. |
| `DeepStreamIngestionManager` | `src/eaigle/ingestion/deepstream_manager.py` | Optional NVIDIA DeepStream backend. `nvurisrcbin → nvstreammux → RGBA pad probe`. Hardware NVDEC decode on GPU. |

**Key design decision:** OpenCV's `cap.read()` is a blocking C call that cannot be awaited. Solution: `loop.run_in_executor(thread_pool, cap.read)` releases the event loop during the blocking call, allowing other coroutines to run concurrently. This enables 50 cameras to run in a single Python process.

**DeepStream backend** (activated by setting `ingestion.backend: deepstream` in config):
- GStreamer pipeline runs in a dedicated GLib thread
- `nvurisrcbin` handles RTSP decode using NVDEC (hardware, zero CPU)
- `nvstreammux` batches multiple streams on the GPU
- Pad probe on `nvstreammux.src` extracts RGBA frames → numpy RGB
- `asyncio.run_coroutine_threadsafe()` bridges GLib thread → asyncio event loop
- Falls back to OpenCV automatically if `gi`/`pyds` not installed

---

### 3.2 Shared Memory Transport

| Component | File | Role |
|---|---|---|
| `write_frame_to_shm` | `src/eaigle/preprocessing/shm_utils.py` | Allocates `/dev/shm/eaigle_{frame_id}`, copies numpy frame bytes. Returns shm name (the "pointer"). |
| `read_frame_from_shm` | `src/eaigle/preprocessing/shm_utils.py` | Maps block, copies frame out, unlinks (frees) the block. |
| `cleanup_stale_shm` | `src/eaigle/preprocessing/shm_utils.py` | Background loop — unlinks blocks older than 10s to prevent memory leaks. |

**Why shared memory instead of Redis for frame data?**

A 1920×1080 BGR frame is ~6 MB. Without shared memory:
- 50 cameras × 10 FPS × 6 MB = **3 GB/s through Redis** → impossible
- Redis would become a bottleneck and RAM would be exhausted

With shared memory:
- Frame pixels stay in `/dev/shm` (kernel RAM, fastest possible access)
- Redis carries only a 200-byte metadata message (the shm name + shape + dtype)
- Preprocessing reads the frame with a memory map — **zero copy**
- After reading, the block is unlinked (freed) immediately

---

### 3.3 Preprocessing Layer

| Component | File | Role |
|---|---|---|
| `PreprocessingWorkerPool` | `src/eaigle/preprocessing/worker_pool.py` | `ProcessPoolExecutor` with 4–8 workers. Each worker process holds a `PreprocessingPipeline` instance. Bounds in-flight tasks to `num_workers × 4` for backpressure. |
| `PreprocessingPipeline` | `src/eaigle/preprocessing/pipeline.py` | Composable chain of ops built from YAML config. Stateless, safe across processes. |
| `ColorConvertOp` | `src/eaigle/preprocessing/ops/color_convert.py` | `cv2.cvtColor(BGR → RGB)` |
| `ResizeOp` | `src/eaigle/preprocessing/ops/resize.py` | `cv2.resize` to 640×640 (YOLO input size) |
| `NormalizeOp` | `src/eaigle/preprocessing/ops/normalize.py` | Divide by 255, convert to float32 |
| `GaussianDenoiseOp` | `src/eaigle/preprocessing/ops/noise_reduction.py` | `cv2.GaussianBlur` |
| `FastNLMeansDenoiseOp` | `src/eaigle/preprocessing/ops/noise_reduction.py` | `cv2.fastNlMeansDenoisingColored` (higher quality, slower) |

**Why ProcessPoolExecutor?**
Python's GIL prevents true CPU parallelism with threads. `ProcessPoolExecutor` spawns separate OS processes, each with their own GIL, enabling genuine parallel preprocessing across CPU cores. A `_pool_init` initializer builds the pipeline once per worker process — not once per frame.

---

### 3.4 Inference Layer

| Component | File | Role |
|---|---|---|
| `DynamicBatcher` | `src/eaigle/inference/batcher.py` | Dual-trigger batch assembly. Fires on: size ≥ `max_size` OR `max_wait_ms` timeout. |
| `InferenceDispatcher` | `src/eaigle/inference/dispatcher.py` | Reads `preprocessed_frames`, feeds batcher. On flush: reads pixels from shm, base64-encodes, POSTs to inference service. Extracts crops for Stage B. |
| Inference Service | `inference_service/app.py` | FastAPI microservice. Stage A: YOLOv8 object detection. Stage B: OCR/classification stub. |

**DynamicBatcher dual-trigger logic:**

```
Frame arrives → add to batch
├── if len(batch) >= max_size (8):   → flush immediately
└── if batch just became non-empty:  → start 50ms countdown timer
    └── when timer expires:          → flush whatever is in batch
```

This ensures:
- Under high load (50 cameras): batches fill quickly → GPU throughput maximized
- Under low load (few cameras): no frame waits more than 50ms

**Without batching:** 50 cameras × 10 FPS = 500 HTTP requests/sec
**With batch size 8:** ≤ 63 HTTP requests/sec, each processing 8 frames in parallel on GPU

---

### 3.5 YOLOv8 Object Detection (Stage A)

Model: **YOLOv8n** (nano) by default — configurable via `YOLO_MODEL` environment variable.

| Model | File Size | Approx Speed (CPU) | mAP50 |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~50ms/frame | 37.3 |
| `yolov8s.pt` | 22 MB | ~90ms/frame | 44.9 |
| `yolov8m.pt` | 52 MB | ~200ms/frame | 50.2 |
| `yolov8l.pt` | 87 MB | ~400ms/frame | 52.9 |
| `yolov8x.pt` | 130 MB | ~700ms/frame | 53.9 |

**COCO → Pipeline label mapping:**

| YOLO COCO Class | Pipeline Label |
|---|---|
| car, bus | vehicle |
| truck | truck |
| person | person |
| bicycle | bicycle |
| motorcycle | motorcycle |
| traffic light | traffic_light |
| stop sign | stop_sign |

All other COCO classes (airplane, boat, dog, etc.) are filtered out as irrelevant to the security/surveillance domain.

---

### 3.6 Aggregation & Fusion Layer

| Component | File | Strategy |
|---|---|---|
| `ResultCollector` | `src/eaigle/aggregation/result_collector.py` | Buffers Stage A + Stage B results per `frame_id`. Marks frame complete when both stages are received. Flushes incomplete frames after 2s timeout. |
| `SpatialNMS` | `src/eaigle/aggregation/fusion_strategies/spatial_nms.py` | Non-Maximum Suppression — removes duplicate bounding boxes with IoU > 0.45. Keeps highest confidence detection. |
| `ConfidenceVoting` | `src/eaigle/aggregation/fusion_strategies/confidence_voting.py` | Filters detections below confidence threshold (0.40). |
| `TemporalSmoother` | `src/eaigle/aggregation/fusion_strategies/temporal_smoother.py` | Per-camera per-label `deque(maxlen=5)`. Label confirmed only if it appears in ≥2 of last 5 frames. Eliminates single-frame false positives. |
| `HypothesisFusion` | `src/eaigle/aggregation/hypothesis_fusion.py` | Orchestrates NMS → voting → smoothing. Produces human-readable `event_description`. |
| `HypothesisStore` | `src/eaigle/aggregation/hypothesis_store.py` | Persists hypothesis to Redis `HSET hypothesis:{id}` (60s TTL) + `LPUSH camera_hypotheses:{cam_id}` (keep 100) + `PUBLISH hypotheses`. |

**Fusion pipeline order:**
```
Raw detections (from Stage A + B)
    │
    ▼
SpatialNMS        → removes: same object detected twice with overlapping boxes
    │
    ▼
ConfidenceVoting  → removes: low-confidence / uncertain detections
    │
    ▼
TemporalSmoother  → removes: single-frame flickers / false positives
    │
    ▼
Confirmed Hypothesis → "Detected at Gate 1 [cam_01]: bus (vehicle, 91%), 3× person"
```

---

### 3.7 Data Models

| Model | File | Fields |
|---|---|---|
| `FrameMetadata` | `src/eaigle/models/frame.py` | `frame_id, camera_id, capture_ts, sequence_num, shm_key, width, height, channels, dtype, pipeline_stage` |
| `BoundingBox` | `src/eaigle/models/detection.py` | `x1, y1, x2, y2` (normalized 0–1). Methods: `.iou()`, `.to_pixel_coords(w,h)` |
| `Detection` | `src/eaigle/models/detection.py` | `stage, label, confidence, bbox, ocr_text, parent_detection_id` |
| `Hypothesis` | `src/eaigle/models/hypothesis.py` | `frame_id, camera_id, detections[], event_description, confidence, pipeline_latency_ms` |
| `CameraConfig` | `src/eaigle/models/hypothesis.py` | `camera_id, rtsp_url, target_fps, width, height, zone` |

---

### 3.8 Transport Layer

| Component | File | Role |
|---|---|---|
| `RedisClient` | `src/eaigle/transport/redis_client.py` | Shared async connection pool. `decode_responses=True`. |
| `StreamProducer` | `src/eaigle/transport/stream_producer.py` | `XADD` with approximate trimming at 500 messages. Idempotent `ensure_group()`. |
| `StreamConsumer` | `src/eaigle/transport/stream_consumer.py` | `XREADGROUP` async iterator. `ack()` after successful processing. |

**Redis Stream names:**

| Stream | Producer | Consumer | Content |
|---|---|---|---|
| `raw_frames:{camera_id}` | CameraWorker | PreprocessingWorkerPool | Raw frame metadata pointer |
| `preprocessed_frames` | PreprocessingWorkerPool | InferenceDispatcher | Preprocessed frame metadata pointer |
| `inference_results` | InferenceDispatcher | ResultCollector | StageResult JSON |
| `hypotheses` (PubSub) | HypothesisStore | Dashboard / Alerts | Final hypothesis events |

---

### 3.9 Observability

| Metric | Type | Description |
|---|---|---|
| `eaigle_frames_ingested_total` | Counter | Frames captured per camera |
| `eaigle_frames_dropped_total` | Counter | Frames dropped (backpressure) |
| `eaigle_preprocessing_latency_ms` | Histogram | Time spent in preprocessing per frame |
| `eaigle_inference_latency_ms` | Histogram | Round-trip time to inference service |
| `eaigle_batch_size` | Histogram | Batch sizes sent to inference |
| `eaigle_detections_total` | Counter | Detections by label |
| `eaigle_hypotheses_total` | Counter | Hypotheses published |
| `eaigle_active_cameras` | Gauge | Currently connected cameras |
| `eaigle_shm_blocks_active` | Gauge | Live shared memory blocks |
| `eaigle_pipeline_latency_ms` | Histogram | End-to-end: capture → hypothesis |
| `eaigle_redis_publish_errors_total` | Counter | Redis write failures |

Prometheus scrapes port **9090**. Grafana can be pointed at this for dashboards.

---

## 4. Data Flow Description

### 4.1 Frame Lifecycle (Single Frame, End-to-End)

```
T=0ms    Camera captures frame
          1920×1080 BGR, uint8, ~6 MB
          │
T=1ms    CameraWorker.run()
          → write_frame_to_shm(frame_id, frame)
          → Creates /dev/shm/eaigle_<uuid> (6MB block)
          → Publishes FrameMetadata to Redis:
            XADD raw_frames:cam_01 * {
              frame_id: "abc-123",
              shm_key:  "eaigle_abc-123",
              width: 1920, height: 1080, channels: 3,
              dtype: "uint8",
              capture_ts: 1709...,
              pipeline_stage: "raw"
            }
          │
T=2ms    PreprocessingWorkerPool
          → XREADGROUP raw_frames:cam_01 (consumer group)
          → Worker process picks up message
          → read_frame_from_shm("eaigle_abc-123", (1080,1920,3), uint8)
            [maps 6MB block, copies out, unlinks — old block freed]
          → Pipeline: BGR→RGB → resize(640,640) → normalize(÷255 → float32)
          → write_frame_to_shm("pp_abc-123", processed_frame)
            [new /dev/shm/eaigle_pp_abc-123 block, 640×640×3×4 = 4.9MB]
          → XACK raw_frames:cam_01 <msg_id>
          → XADD preprocessed_frames * {
              shm_key: "eaigle_pp_abc-123",
              width: 640, height: 640,
              dtype: "float32", ...
            }
          │
T=5ms    InferenceDispatcher
          → XREADGROUP preprocessed_frames
          → Frame enters DynamicBatcher queue
          │
          (waits for batch: size=8 OR 50ms timeout)
          │
T=55ms   DynamicBatcher flushes batch of N frames
          → For each frame: read_frame_from_shm → base64 encode
          → HTTP POST http://inference:8001/detect/primary
            body: {batch_id: "...", frames: [{frame_id, data_b64, shape, dtype}, ...]}
          │
T=75ms   YOLOv8 runs on batch (GPU or CPU)
          → Returns:
            [{label:"bus",    conf:0.91, bbox:{x1:0.1,y1:0.2,x2:0.8,y2:0.9}},
             {label:"person", conf:0.87, bbox:{...}},
             {label:"person", conf:0.79, bbox:{...}}]
          │
T=76ms   Dispatcher extracts crops for Stage B
          → For each "vehicle"/"person" detection:
            crop = frame[y1:y2, x1:x2]
          → HTTP POST /detect/secondary
            body: {frames: [{data_b64: crop_b64, label_hint: "vehicle", ...}]}
          → Secondary returns: {label:"license_plate", ocr_text:"ABC123", conf:0.88}
          │
T=80ms   Dispatcher publishes StageResults to Redis:
          XADD inference_results * {
            capture_ts: "1709...",
            result_json: "{stage: primary, detections: [...]}"
          }
          XADD inference_results * {
            capture_ts: "1709...",
            result_json: "{stage: secondary, detections: [...]}"
          }
          │
T=81ms   ResultCollector receives Stage A + Stage B
          → FrameAccumulator groups by frame_id
          → When both stages present → triggers HypothesisFusion
          │
T=82ms   HypothesisFusion
          1. SpatialNMS:        [bus(0.91), person(0.87), person(0.79)] → same (no overlap)
          2. ConfidenceVoting:  all > 0.40 → all pass
          3. TemporalSmoother:  "bus" seen in 3 of last 5 frames → confirmed
          → event_description: "Detected at Gate 1 [cam_01]: bus (vehicle, 91%), 2× person"
          │
T=83ms   HypothesisStore
          → HSET hypothesis:xyz {frame_id, camera_id, detections, event_description, ...}
             (TTL: 60 seconds)
          → LPUSH camera_hypotheses:cam_01 "xyz"  (keep latest 100)
          → PUBLISH hypotheses {hypothesis JSON}
          │
          ✓ End-to-end latency: ~83ms
          ✓ Event available to all downstream subscribers
```

### 4.2 Concurrency Model

```
Main Process (asyncio event loop)
├── CameraWorker × 50          (coroutines, I/O-bound)
│   └── ThreadPoolExecutor     (for blocking cv2.read calls)
│
├── PreprocessingWorkerPool    (asyncio consumer)
│   └── ProcessPoolExecutor    (4–8 processes, CPU-bound)
│       ├── Worker 0: pipeline init + frame processing
│       ├── Worker 1: pipeline init + frame processing
│       └── ...
│
├── InferenceDispatcher        (coroutine, I/O-bound)
│   └── DynamicBatcher         (size OR time trigger)
│       └── httpx.AsyncClient  (HTTP to inference service)
│
├── ResultCollector            (coroutine)
│   └── HypothesisFusion       (in-process, fast)
│       └── HypothesisStore    (Redis writes)
│
└── Background tasks
    ├── SHM cleanup loop       (every 10s)
    └── Prometheus metrics     (port 9090)
```

### 4.3 Backpressure Mechanism

If preprocessing falls behind ingestion:
1. `PreprocessingWorkerPool` bounds in-flight tasks to `num_workers × 4`
2. When limit reached → camera workers check `BACKPRESSURE_THRESHOLD` (300 messages in Redis stream)
3. If stream is full → `CameraWorker` skips publishing (drops frame rather than OOMing)
4. `eaigle_frames_dropped_total` Prometheus counter increments

---

## 5. Deployment Strategy

### 5.1 Local Development

Prerequisites: Docker (for Redis), Python 3.11+

```bash
# Clone and install
cd EAIGLE_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install ultralytics matplotlib

# Terminal 1 — Redis
docker run -d --name eaigle-redis -p 6379:6379 redis:7-alpine \
  --maxmemory 256mb --maxmemory-policy allkeys-lru

# Terminal 2 — Inference service (downloads YOLOv8n on first run ~6MB)
uvicorn inference_service.app:app --host 0.0.0.0 --port 8001

# Terminal 3 — Camera simulator (synthetic frames with real image if available)
#   Place any JPEG at /tmp/sample_frame.jpg for real YOLO detections
python scripts/simulate_cameras.py --cameras 5 --fps 10

# Terminal 4 — Main pipeline
python -m eaigle.app --config configs/pipeline.yaml

# Run tests
pytest tests/unit/ -v
```

**Expected output (with sample image):**
```
INFO  eaigle.aggregation.hypothesis_store [cam_01] Detected at Gate 1 (cam_01):
      bus (vehicle, 91%), person (person, 87%), person (person, 79%)
      (conf=0.86, latency=83ms)
```

---

### 5.2 Docker Compose (Staging / Demo)

```bash
cd docker
docker compose up
```

**Services:**

| Service | Image | Port | Notes |
|---|---|---|---|
| `redis` | `redis:7-alpine` | 6379 | 256MB maxmemory, LRU eviction |
| `inference` | `Dockerfile.inference` | 8001 | YOLOv8n, 2 uvicorn workers |
| `pipeline` | `Dockerfile.pipeline` | — | Main asyncio app |
| `simulator` | same as pipeline | — | 5 synthetic cameras @ 10 FPS |
| `prometheus` | `prom/prometheus:latest` | 9090 | Scrapes pipeline metrics |

**DeepStream variant** (requires NVIDIA GPU + drivers):
```bash
docker compose --profile deepstream up
```

**Useful commands:**
```bash
# View logs
docker compose logs -f pipeline

# Check Redis stream depth
docker exec eaigle-redis redis-cli XLEN preprocessed_frames

# Stop everything
docker compose down -v
```

---

### 5.3 Kubernetes (Production)

**Recommended cluster topology for 50 cameras:**

```
Namespace: eaigle
│
├── redis-cluster (StatefulSet)
│   ├── 3 replicas with Redis Sentinel for HA
│   └── PersistentVolumeClaim: 10Gi SSD
│
├── inference-service (Deployment)
│   ├── replicas: 2–8 (auto-scaled)
│   ├── nodeSelector: cloud.google.com/gke-accelerator: nvidia-tesla-t4
│   ├── resources.limits: {nvidia.com/gpu: 1, memory: 8Gi}
│   └── HPA: scale on GPU utilization > 70%
│
├── pipeline (Deployment)
│   ├── replicas: 5 (10 cameras per pod)
│   ├── env: CAMERAS_PER_INSTANCE=10
│   ├── volumes: {medium: Memory, sizeLimit: 2Gi}  ← /dev/shm
│   └── resources: {cpu: 4, memory: 4Gi}
│
└── simulator (Job) — only for testing
```

**Key Kubernetes manifest snippets:**

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: eaigle
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
        - name: inference
          image: eaigle/inference:latest
          ports:
            - containerPort: 8001
          env:
            - name: YOLO_MODEL
              value: "yolov8s.pt"
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "8Gi"
            requests:
              cpu: "2"
              memory: "4Gi"
---
# inference-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: eaigle
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# pipeline-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline
  namespace: eaigle
spec:
  replicas: 5
  template:
    spec:
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
      containers:
        - name: pipeline
          image: eaigle/pipeline:latest
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
          env:
            - name: REDIS_URL
              value: "redis://redis-svc:6379/0"
          resources:
            requests:
              cpu: "4"
              memory: "4Gi"
```

**Scaling strategy:**

| Component | Scale Unit | Trigger |
|---|---|---|
| `pipeline` pods | +1 pod per 10 cameras | Fixed: cameras ÷ 10 |
| `inference-service` | +1 pod | GPU utilization > 70% |
| `redis` | Fixed 3 replicas | HA requirement (Sentinel) |

**Important note:** POSIX shared memory is per-node. The `pipeline` pod (CameraWorker + PreprocessingWorkerPool) must run on a single pod since both stages use the same `/dev/shm`. The shm volume above uses `emptyDir: {medium: Memory}` which is pod-local RAM.

For **multi-node** inference scaling: the inference service is already network-separated (HTTP), so it naturally scales horizontally. Only the preprocessing→dispatcher path needs to be co-located.

---

## 6. Configuration Reference

**`configs/pipeline.yaml`** — all tunable parameters:

```yaml
# ── Ingestion ─────────────────────────────────────────────────────────────────
ingestion:
  backend: opencv          # "opencv" | "deepstream"
  deepstream:
    output_width: 1280
    output_height: 720

cameras:
  streams:
    - id: cam_01
      rtsp_url: rtsp://user:pass@192.168.1.10:554/stream
      target_fps: 10
      width: 1920
      height: 1080
      buffer_size: 1         # cv2 capture buffer (1 = always freshest frame)
      reconnect_delay_s: 2.0
      max_consecutive_failures: 30
      zone: "Gate 1"         # Used in event_description

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocessing:
  num_workers: 4             # ProcessPoolExecutor size (match CPU cores)
  ops:
    - type: color_convert
      params: {src: BGR, dst: RGB}
    - type: resize
      params: {width: 640, height: 640}
    - type: normalize        # ÷255 → float32 (skip if using YOLO's internal norm)
    # - type: gaussian_denoise
    #   params: {ksize: 3}
    # - type: nlm_denoise
    #   params: {h: 10, template_window_size: 7, search_window_size: 21}

# ── Inference ─────────────────────────────────────────────────────────────────
inference:
  service_url: http://localhost:8001
  batch_size: 8              # Max frames per batch
  batch_timeout_ms: 50       # Max wait before flushing partial batch

# ── Aggregation ───────────────────────────────────────────────────────────────
aggregation:
  nms_iou_threshold: 0.45    # Boxes with IoU > this are merged
  confidence_threshold: 0.40  # Detections below this are dropped
  temporal_window: 5          # Number of frames to consider
  temporal_min_presence: 2    # Min appearances in window to confirm label

# ── Redis ─────────────────────────────────────────────────────────────────────
redis:
  url: redis://localhost:6379/0
  stream_maxlen: 500          # Approximate max messages per stream

# ── Observability ─────────────────────────────────────────────────────────────
observability:
  log_level: INFO
  metrics_port: 9090
```

---

## 7. How to Run

### Quick Start (3 commands)

```bash
# 1. Start Redis
docker run -d --name eaigle-redis -p 6379:6379 redis:7-alpine

# 2. Start inference service
.venv/bin/uvicorn inference_service.app:app --port 8001

# 3. Run simulator + pipeline (two terminals)
.venv/bin/python scripts/simulate_cameras.py --cameras 5 --fps 10
.venv/bin/python -m eaigle.app --config configs/pipeline.yaml
```

### Test Inference Service Directly

```bash
# Open API docs in browser
open http://localhost:8001/docs

# Or test from terminal
.venv/bin/python - <<'EOF'
import requests, numpy as np, base64, uuid, cv2

img = cv2.imread("/tmp/sample_frame.jpg")
img = cv2.resize(img, (640, 640))

resp = requests.post("http://localhost:8001/detect/primary", json={
    "batch_id": str(uuid.uuid4()),
    "frames": [{
        "frame_id": str(uuid.uuid4()),
        "camera_id": "test_cam",
        "data_b64": base64.b64encode(img.tobytes()).decode(),
        "shape": list(img.shape),
        "dtype": "uint8",
    }]
})
for det in resp.json()["results"]:
    print(f"{det['label']:12s} conf={det['confidence']:.2f}  bbox={det['bbox']}")
EOF
```

### Run Unit Tests

```bash
.venv/bin/pytest tests/unit/ -v
# 31 tests: preprocessing, batcher, fusion, models, shm
```

### Switch to Larger YOLO Model

```bash
YOLO_MODEL=yolov8s.pt .venv/bin/uvicorn inference_service.app:app --port 8001
```

### Use Real RTSP Camera

Edit `configs/pipeline.yaml`:
```yaml
cameras:
  streams:
    - id: cam_01
      rtsp_url: rtsp://admin:password@192.168.1.100:554/live/ch0
      target_fps: 15
      zone: "Entrance"
```

---

## File Structure

```
EAIGLE_AI/
├── src/eaigle/
│   ├── app.py                          # Main entry point
│   ├── ingestion/
│   │   ├── camera_worker.py            # Single RTSP camera coroutine
│   │   ├── camera_manager.py           # Supervises N cameras
│   │   └── deepstream_manager.py       # DeepStream GPU backend
│   ├── preprocessing/
│   │   ├── shm_utils.py                # POSIX shared memory helpers
│   │   ├── pipeline.py                 # Composable op chain
│   │   ├── worker_pool.py              # ProcessPoolExecutor consumer
│   │   └── ops/                        # Individual preprocessing ops
│   ├── inference/
│   │   ├── batcher.py                  # DynamicBatcher
│   │   └── dispatcher.py              # Reads shm, calls inference API
│   ├── aggregation/
│   │   ├── result_collector.py         # Groups Stage A + B per frame
│   │   ├── hypothesis_fusion.py        # Orchestrates fusion pipeline
│   │   ├── hypothesis_store.py         # Redis persistence + PubSub
│   │   └── fusion_strategies/
│   │       ├── spatial_nms.py          # IoU-based deduplication
│   │       ├── confidence_voting.py    # Score threshold filter
│   │       └── temporal_smoother.py   # 5-frame sliding window
│   ├── models/
│   │   ├── frame.py                    # FrameMetadata dataclass
│   │   ├── detection.py                # BoundingBox, Detection
│   │   └── hypothesis.py              # Hypothesis, CameraConfig
│   ├── transport/
│   │   ├── redis_client.py             # Async Redis connection pool
│   │   ├── stream_producer.py          # XADD wrapper
│   │   └── stream_consumer.py          # XREADGROUP async iterator
│   └── observability/
│       ├── metrics.py                  # Prometheus metrics definitions
│       └── logging_config.py          # Structured JSON logging
├── inference_service/
│   └── app.py                          # FastAPI: YOLOv8 + OCR stub
├── configs/
│   ├── pipeline.yaml                   # Main config
│   └── pipeline-deepstream.yaml        # DeepStream variant config
├── docker/
│   ├── docker-compose.yml              # Full stack
│   ├── Dockerfile.pipeline             # OpenCV backend image
│   ├── Dockerfile.inference            # YOLOv8 inference image
│   ├── Dockerfile.pipeline-deepstream  # DeepStream image
│   └── prometheus.yml                  # Prometheus scrape config
├── scripts/
│   ├── simulate_cameras.py             # Synthetic RTSP simulator
│   └── draw_architecture.py           # Generates docs/architecture.png
├── tests/unit/
│   ├── test_preprocessing.py
│   ├── test_batcher.py
│   ├── test_fusion.py
│   ├── test_models.py
│   └── test_shm.py
├── docs/
│   ├── architecture.png                # Generated diagram
│   ├── architecture.pdf                # Generated diagram (PDF)
│   └── report.md                       # This file
└── pyproject.toml                      # Package definition + dependencies
```

---

## 8. Technical Design Document

### 8.1 Concurrency Model

The pipeline uses a **layered concurrency model** — each layer uses the right primitive for its workload type:

| Layer | Primitive | Reason |
|---|---|---|
| Ingestion (50 cameras) | `asyncio` coroutines + `ThreadPoolExecutor` | RTSP read is blocking I/O; thread pool unblocks the event loop |
| Preprocessing | `ProcessPoolExecutor` (4–8 workers) | CPU-bound (resize, normalize); bypasses Python GIL with separate processes |
| Inference dispatch | `asyncio` + `httpx.AsyncClient` | Network I/O to inference service; async avoids blocking |
| Aggregation | `asyncio` coroutine | Light computation; no blocking calls |
| DeepStream (optional) | GLib thread + `asyncio.run_coroutine_threadsafe` | GStreamer runs in GLib main loop; bridge to asyncio via thread-safe call |

**Why not threads everywhere?**
Python's Global Interpreter Lock (GIL) ensures only one thread executes Python bytecode at a time. For CPU-bound work (frame resize, normalization), threads give no speedup — only `ProcessPoolExecutor` (true separate OS processes) achieves real parallelism.

**Why not processes everywhere?**
Processes have high startup cost and IPC overhead. For I/O-bound work (RTSP reads, Redis, HTTP), a single asyncio event loop handles thousands of concurrent operations efficiently via the OS `epoll`/`kqueue` multiplexer.

```
Single OS Process (Main)
│
├── asyncio event loop
│   ├── CameraWorker × 50         [coroutines, non-blocking]
│   │   └── ThreadPoolExecutor    [cv2.read() offloaded here]
│   ├── PreprocessingWorkerPool   [coroutine, submits to processes]
│   │   └── ProcessPoolExecutor   [4-8 OS processes, true parallel]
│   ├── InferenceDispatcher       [coroutine, async HTTP]
│   ├── ResultCollector           [coroutine]
│   └── Background tasks         [shm cleanup, metrics]
```

---

### 8.2 Queueing Strategy

The pipeline uses **Redis Streams** as durable, ordered queues between every stage. Redis Streams were chosen over Kafka for simplicity (no ZooKeeper, lower ops overhead) while providing equivalent guarantees at this scale.

**Three queues in the pipeline:**

```
[Ingestion] ──XADD──► raw_frames:{cam_id}   ──XREADGROUP──► [Preprocessing]
[Preprocessing] ──XADD──► preprocessed_frames ──XREADGROUP──► [Dispatcher]
[Dispatcher] ──XADD──► inference_results       ──XREADGROUP──► [Collector]
[Collector] ──PUBLISH──► hypotheses (PubSub)   ──SUBSCRIBE──► [Downstream]
```

**Consumer Groups** (`XREADGROUP`):
- Multiple worker processes can consume from the same stream concurrently
- Each message is delivered to exactly one consumer in the group
- Messages are acknowledged (`XACK`) only after successful processing
- Unacknowledged messages are redeliverable (PEL — Pending Entry List)

**Stream trimming:**
```python
STREAM_MAXLEN = 500   # approximate, O(1) trimming
XADD stream MAXLEN ~ 500 * {message}
```
Prevents unbounded memory growth if consumers fall behind.

**DynamicBatcher queue** (in-process, before inference):
- Not a Redis queue — in-memory `asyncio.Queue` inside the dispatcher
- Purpose: accumulate frames into GPU-efficient batches before the HTTP call
- Dual trigger: `max_size=8` (throughput) OR `max_wait_ms=50` (latency)

---

### 8.3 Back-Pressure Handling

Back-pressure prevents fast producers from overwhelming slow consumers and causing out-of-memory crashes.

**Level 1 — In-process (Preprocessing → Dispatcher):**
```python
# worker_pool.py
MAX_INFLIGHT = num_workers * 4   # e.g. 8 workers → 32 max in-flight tasks

if self._inflight >= MAX_INFLIGHT:
    await asyncio.sleep(0.01)    # yield; do not submit more work
```
Effect: if preprocessing workers are busy, no new frames are read from Redis.

**Level 2 — Redis stream depth (Ingestion → Preprocessing):**
```python
# camera_worker.py
BACKPRESSURE_THRESHOLD = 300    # max messages in raw_frames stream

stream_len = await redis.xlen(f"raw_frames:{camera_id}")
if stream_len > BACKPRESSURE_THRESHOLD:
    # Skip publishing this frame — drop rather than OOM
    metrics.frames_dropped.inc()
    continue
```
Effect: if preprocessing is slow, camera workers voluntarily skip frames. Freshness (latest frame) is prioritized over completeness.

**Level 3 — Redis memory limit:**
```
Redis config: maxmemory 256mb, maxmemory-policy allkeys-lru
```
Effect: Redis itself evicts old stream entries under memory pressure. This is the last resort.

**Back-pressure flow:**
```
GPU slow? → inference HTTP takes longer → batcher fills up → dispatcher slow
    → preprocessed_frames stream grows → preprocessing slows down
    → raw_frames stream grows → camera workers start dropping frames
    → system stabilizes at sustainable throughput
```

---

### 8.4 Failure Recovery

**Camera disconnection:**
```python
# camera_worker.py
consecutive_failures = 0
while not stop_event.is_set():
    ret, frame = cap.read()
    if not ret:
        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:  # default: 30
            cap.release()
            cap = None   # triggers reconnect on next iteration
    else:
        consecutive_failures = 0
```
- Auto-reconnects after `max_consecutive_failures` failures
- `reconnect_delay_s=2.0` prevents reconnect storms

**Worker process crash (Preprocessing):**
```python
# camera_manager.py — supervision loop
while not stop_event.is_set():
    done, pending = await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
    for task in done:
        exc = task.exception()
        if exc:
            logger.error("Camera task crashed: %s — restarting", exc)
            new_task = asyncio.create_task(worker.run())
            tasks.add(new_task)
    await asyncio.sleep(2.0)
```
- CameraManager supervision loop restarts any crashed camera task
- ProcessPoolExecutor automatically replaces crashed worker processes

**Redis connection lost:**
```python
# redis_client.py
client = aioredis.from_url(url,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,       # auto-retry
    health_check_interval=30,    # detect stale connections
)
```
- `retry_on_timeout=True` handles transient network hiccups
- All pipeline components share the same connection pool

**Inference service unavailable:**
```python
# dispatcher.py
try:
    resp = await self._client.post(url, json=payload, timeout=5.0)
    resp.raise_for_status()
    return resp.json()
except Exception as exc:
    logger.error("Inference call failed: %s", exc)
    return None   # batch is skipped; frames dropped; pipeline continues
```
- Inference failure is non-fatal — pipeline continues processing
- Frames in the batch are dropped (not requeued) to avoid stale data accumulation

**Stale shared memory blocks:**
```python
# Background task in app.py, runs every 10s
cleaned = cleanup_stale_shm(prefix="eaigle_", max_age_s=10.0)
```
- Scans `/dev/shm` and unlinks blocks older than 10s
- Prevents memory leak if a consumer crashes before unlinking a block

**Unacknowledged Redis messages (PEL recovery):**
- If a preprocessing worker crashes mid-processing, its message stays in the PEL
- On restart, the worker calls `XAUTOCLAIM` to recover messages idle > 30s
- Guarantees at-least-once processing even across crashes

---

### 8.5 Scaling Strategy

#### Horizontal Scaling (more cameras)

| Camera Count | Pipeline Pods | Preprocessing Workers | Inference Replicas |
|---|---|---|---|
| 5 cameras | 1 pod | 2 workers | 1 replica |
| 50 cameras | 5 pods (10 cam/pod) | 4 workers/pod | 2 replicas |
| 200 cameras | 20 pods (10 cam/pod) | 8 workers/pod | 4–8 replicas |

Each pipeline pod is independent — no shared state except Redis (which is the designed coordination point). Adding cameras = adding pods, not modifying existing ones.

#### Vertical Scaling (better hardware)

| Resource | Effect |
|---|---|
| More CPU cores | Increase `num_workers` in preprocessing pool |
| More RAM | Increase Redis `maxmemory`; allow larger `/dev/shm` |
| GPU upgrade | Larger YOLO model (`yolov8x.pt`); larger batch size |
| Faster NVMe | `/dev/shm` is RAM, not disk — no disk I/O in hot path |

#### Redis scaling (> 50 cameras)

At 200 cameras × 10 FPS = 2,000 metadata messages/sec through Redis. Redis handles 100k+ ops/sec on a single node — this is not a bottleneck. If needed:
- Redis Cluster: shard streams by camera group
- Redis Sentinel: HA failover (already in K8s config)

---

### 8.6 GPU Utilization Strategy

**Problem:** GPUs are most efficient when processing large contiguous batches. A single frame request wastes most of the GPU's parallel compute capacity.

**Solution — DynamicBatcher:**
```
Frame 1 arrives at T=0ms  → batch starts, 50ms timer starts
Frame 2 arrives at T=5ms  → added to batch
...
Frame 8 arrives at T=20ms → batch full → flush immediately (size trigger)

OR

Frame 1 arrives at T=0ms  → batch starts
Frame 2 arrives at T=10ms → added to batch
T=50ms                    → timer fires → flush with 2 frames (time trigger)
```

**Batch size vs latency trade-off:**

| Batch Size | GPU Utilization | Max Added Latency |
|---|---|---|
| 1 (no batching) | ~5% | 0ms |
| 4 | ~40% | 50ms |
| 8 (default) | ~75% | 50ms |
| 16 | ~90% | 50ms |
| 32 | ~95% | 50ms |

**YOLOv8 GPU inference time by batch size** (T4 GPU, 640×640):

| Batch | Time | Throughput |
|---|---|---|
| 1 | 8ms | 125 FPS |
| 8 | 15ms | 533 FPS |
| 16 | 25ms | 640 FPS |
| 32 | 45ms | 711 FPS |

For 50 cameras × 10 FPS = 500 frames/sec → batch size 8 → 63 batches/sec → T4 handles this in ~15ms/batch = 945ms of GPU time per second → **~95% GPU utilization**.

**DeepStream backend** (for maximum GPU efficiency):
- `nvurisrcbin` + NVDEC: hardware H.264/H.265 decode on GPU, 0% CPU
- `nvstreammux`: batches N camera streams directly on GPU memory
- Frames never leave GPU memory between decode → inference → result
- Eliminates CPU↔GPU memory copy overhead (~6ms per frame at 1080p)

---

### 8.7 Latency Estimates

**Per-frame end-to-end latency breakdown:**

| Stage | Duration | Notes |
|---|---|---|
| RTSP capture → shm write | 1ms | Memory copy only |
| Redis XADD (ingestion) | <1ms | Single round-trip |
| Redis XREADGROUP (preprocessing) | <1ms | |
| Preprocessing (color+resize+normalize) | 3–8ms | CPU, parallelized |
| Redis XADD (preprocessed) | <1ms | |
| Batcher wait | 0–50ms | Worst case: 50ms timeout |
| shm read + base64 encode | 2ms | |
| HTTP POST to inference | 1ms | LAN round-trip |
| YOLOv8n inference (batch=8, CPU) | 50ms | No GPU |
| YOLOv8n inference (batch=8, T4 GPU) | 15ms | With GPU |
| Crop extraction + secondary stage | 5ms | |
| Redis XADD (results) | <1ms | |
| Aggregation + fusion | 1ms | In-process |
| Redis HSET + PUBLISH | <1ms | |

**Total estimated latency:**

| Hardware | Best Case | Typical | Worst Case |
|---|---|---|---|
| CPU only | 65ms | 100ms | 200ms |
| With GPU (T4) | 30ms | 50ms | 100ms |
| With DeepStream + GPU | 25ms | 40ms | 80ms |

---

### 8.8 Throughput Estimates

**Single pipeline pod (10 cameras, CPU only):**
```
Input:  10 cameras × 10 FPS = 100 frames/sec
Preprocessing: 4 workers × (1000ms / 8ms per frame) = 500 frames/sec capacity
Inference: batch=8, 100ms/batch (CPU) = 80 frames/sec capacity
Bottleneck: inference (CPU) → ~80 FPS total
```

**Single pipeline pod (10 cameras, T4 GPU):**
```
Input:  10 cameras × 10 FPS = 100 frames/sec
Preprocessing: 500 frames/sec capacity
Inference: batch=8, 15ms/batch (GPU) = 533 frames/sec capacity
Bottleneck: ingestion → handles load easily, ~100 FPS
```

**50 cameras system (5 pods + 2 GPU inference replicas):**
```
Input:  50 cameras × 10 FPS = 500 frames/sec
Inference capacity: 2 replicas × 533 FPS = 1,066 frames/sec
Redis throughput: ~2,000 ops/sec (well within Redis 100k/sec limit)
Result: system has 2× headroom — can absorb burst to 20 FPS per camera
```

**200 cameras system (20 pods + 4–8 GPU inference replicas):**
```
Input:  200 cameras × 10 FPS = 2,000 frames/sec
Inference capacity: 8 replicas × 533 FPS = 4,264 frames/sec
Redis throughput: ~8,000 ops/sec (still well within limits)
Result: comfortable margin; can run at 20 FPS or higher-resolution models
```

---

## 9. Code — Prototype Implementation

### 9.1 RTSP Streaming — Multiple Cameras

Implemented in `src/eaigle/ingestion/camera_worker.py` and `camera_manager.py`.

**Real RTSP (production):**
```python
class CameraWorker:
    async def run(self):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            # Blocking cv2.read() offloaded to thread pool
            ret, frame = await loop.run_in_executor(
                self._executor, self._read_frame
            )
            if ret:
                shm_key = await loop.run_in_executor(
                    self._executor, write_frame_to_shm, frame_id, frame
                )
                await self._producer.publish(
                    stream_name=f"raw_frames:{camera_id}",
                    message=metadata.to_dict(),
                )
```

**Mock cameras (testing, no RTSP server needed):**
```bash
# Generates synthetic frames: colored noise + timestamp + moving rectangle
# Optionally loads a real image from /tmp/sample_frame.jpg
python scripts/simulate_cameras.py --cameras 5 --fps 10 --duration 60
```
The simulator publishes directly to Redis Streams, bypassing OpenCV capture.
It is **fully compatible with the rest of the pipeline** — preprocessing, inference, and aggregation receive the same message format as real cameras.

---

### 9.2 Preprocessing Module

Implemented in `src/eaigle/preprocessing/`.

```python
# Composable pipeline — built from YAML config
pipeline = PreprocessingPipeline(ops=[
    ColorConvertOp(src="BGR", dst="RGB"),
    ResizeOp(width=640, height=640),
    NormalizeOp(),   # ÷255 → float32
])

# Runs in isolated process (ProcessPoolExecutor)
def _process_frame(shm_key, shape, dtype):
    frame = read_frame_from_shm(shm_key, shape, dtype)  # zero-copy read
    result = pipeline.run(frame)                          # CPU processing
    return result, elapsed_ms
```

Key properties:
- **Stateless**: same function runs safely in any worker process
- **Composable**: ops are added/removed via YAML — no code changes
- **Zero-copy read**: memory-mapped shm block, not deserialized from Redis

---

### 9.3 Inference Service Stub

Implemented in `inference_service/app.py`.

The service has two modes:

**Mode 1 — YOLOv8 (real inference, default):**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # downloads 6MB on first run

@app.post("/detect/primary")
async def detect_primary(req: BatchRequest):
    for frame_payload in req.frames:
        img = _decode_frame(frame_payload)  # base64 → uint8 numpy
        results = model.predict(img, verbose=False, conf=0.35)
        # returns real bboxes: [{label:"bus", conf:0.91, bbox:{...}}, ...]
```

**Mode 2 — Stub fallback** (if `ultralytics` not installed):
```python
def _stub_primary_detect(frame_id):
    return [DetectionResult(
        label=random.choice(["vehicle", "person", "truck"]),
        confidence=random.uniform(0.55, 0.99),
        bbox=BBoxResult(x1=random.uniform(0,0.5), ...),
    ) for _ in range(random.randint(0, 3))]
```

Both modes return identical JSON schema — the pipeline cannot tell the difference.

---

### 9.4 End-to-End Data Flow

```
simulate_cameras.py          inference_service/app.py
        │                            ▲
        │ XADD raw_frames:*          │ HTTP POST /detect/primary
        ▼                            │ HTTP POST /detect/secondary
   Redis Streams                     │
        │                            │
        │ XREADGROUP                 │
        ▼                            │
  worker_pool.py               dispatcher.py
  (preprocessing)  ──────────► (batching + dispatch)
        │                            │
        │ XADD preprocessed_frames   │ XADD inference_results
        ▼                            ▼
   Redis Streams              Redis Streams
                                     │
                                     │ XREADGROUP
                                     ▼
                             result_collector.py
                             hypothesis_fusion.py
                                     │
                                     │ HSET + PUBLISH hypotheses
                                     ▼
                                Redis PubSub
                                     │
                             downstream consumers
```

All stages run concurrently in the same process via `asyncio.gather()`:
```python
# app.py
await asyncio.gather(
    camera_manager.run(),          # ingestion
    preprocessing_pool.run(),      # preprocessing
    inference_dispatcher.run(),    # inference
    result_collector.run(),        # aggregation
    _shm_cleanup_loop(),           # maintenance
)
```

---

### 9.5 Logging & Metrics

**Structured logging** — every log line includes context:
```python
logger.info(
    "Primary batch %s: %d frames → %d detections in %.1fms",
    batch_id, batch_size, len(results), latency_ms,
)
# Output:
# 2026-03-04 10:02:05 INFO eaigle.inference.dispatcher
#   Primary batch abc-123: 8 frames → 12 detections in 47.3ms
```

**Prometheus metrics** (scraped at `:9090/metrics`):
```python
# metrics.py — 11 metrics total
frames_ingested   = Counter("eaigle_frames_ingested_total", ["camera_id"])
frames_dropped    = Counter("eaigle_frames_dropped_total", ["camera_id"])
preproc_latency   = Histogram("eaigle_preprocessing_latency_ms", buckets=[...])
inference_latency = Histogram("eaigle_inference_latency_ms", buckets=[...])
batch_size_hist   = Histogram("eaigle_batch_size", buckets=[1,2,4,8,16,32])
detections_total  = Counter("eaigle_detections_total", ["label"])
hypotheses_total  = Counter("eaigle_hypotheses_total", ["camera_id"])
active_cameras    = Gauge("eaigle_active_cameras")
shm_blocks_active = Gauge("eaigle_shm_blocks_active")
pipeline_latency  = Histogram("eaigle_pipeline_latency_ms", buckets=[...])
redis_errors      = Counter("eaigle_redis_publish_errors_total")
```

Example Prometheus queries for Grafana dashboards:
```promql
# Frames per second per camera
rate(eaigle_frames_ingested_total[1m])

# 95th percentile end-to-end latency
histogram_quantile(0.95, eaigle_pipeline_latency_ms_bucket)

# Drop rate (should be ~0 under normal load)
rate(eaigle_frames_dropped_total[5m])

# Inference throughput
rate(eaigle_detections_total[1m])
```

---

## 10. Performance Considerations

### 10.1 Scaling from 5 → 50 → 200 Cameras

**5 cameras (single developer machine):**
- 1 pipeline process, 2 preprocessing workers
- CPU inference (YOLOv8n): 50ms/batch → 5 cameras × 10 FPS = 50 FPS total, handled easily
- Memory: 5 × 6MB shm + Redis ≈ 100MB total
- No GPU required

**50 cameras (small server or cloud VM):**
- 5 pipeline pods (10 cameras each), 4 workers/pod
- 1–2 GPU inference replicas (T4 GPU, $0.35/hr on GCP)
- Redis: single node with 512MB memory limit
- Memory: 50 × 6MB shm = 300MB + 512MB Redis = ~1GB total

**200 cameras (production cluster):**
- 20 pipeline pods, 8 preprocessing workers/pod (160 total parallel preprocessors)
- 4–8 GPU inference replicas with HPA auto-scaling
- Redis Sentinel (3 nodes) for HA
- Network: 200 × 6MB × 10 FPS = 12 GB/s between cameras and pipeline pods
  → Must co-locate pipeline pods with cameras or use 25GbE NICs
  → With DeepStream: NVDEC handles network → GPU directly, bypassing CPU
- `/dev/shm`: each pod uses up to 640KB/frame × 10 frames in-flight = 64MB
  → Well within 2GB pod limit

**Scaling table summary:**

| Metric | 5 cameras | 50 cameras | 200 cameras |
|---|---|---|---|
| Pipeline pods | 1 | 5 | 20 |
| GPU replicas | 0 (CPU OK) | 2 | 4–8 |
| Redis nodes | 1 | 1 | 3 (Sentinel) |
| CPU cores needed | 4 | 40 | 160 |
| GPU needed | None | 2× T4 | 4–8× T4 |
| RAM needed | 4GB | 20GB | 80GB |

---

### 10.2 Handling GPU Bottlenecks

**Detect GPU bottleneck:**
```promql
# If inference latency P95 > 100ms → GPU is bottlenecked
histogram_quantile(0.95, eaigle_inference_latency_ms_bucket) > 100

# If preprocessed_frames stream depth grows → inference falling behind
# (monitor via Redis XLEN preprocessed_frames)
```

**Mitigation strategies (in order of cost):**

1. **Increase batch size** (free):
   ```yaml
   # pipeline.yaml
   inference:
     batch_size: 16   # was 8 — doubles GPU utilization
   ```

2. **Use smaller YOLO model** (free):
   ```bash
   YOLO_MODEL=yolov8n.pt   # nano: 50ms/batch → switch from yolov8m
   ```

3. **Add inference replicas** (horizontal scale):
   ```bash
   kubectl scale deployment inference-service --replicas=4
   ```
   Redis streams fan out to multiple inference pods automatically.

4. **Enable TensorRT** (3–5× GPU speedup, one-time export):
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="engine")   # converts to TensorRT .engine file
   # Then: YOLO_MODEL=yolov8n.engine
   ```

5. **Switch to DeepStream** (NVDEC + GPU pipeline, eliminates CPU decode):
   ```yaml
   ingestion:
     backend: deepstream
   ```

---

### 10.3 Model Batching Strategies

Three batching strategies compared:

**Strategy A — Fixed size batch (simple):**
```
Wait until exactly N frames arrive, then flush.
Pros: predictable GPU load
Cons: high latency when cameras are idle (wait forever for frame N)
```

**Strategy B — Fixed time window:**
```
Every T milliseconds, flush whatever accumulated.
Pros: bounded latency
Cons: under-utilizes GPU when traffic is low
```

**Strategy C — Dual trigger (implemented — DynamicBatcher):**
```
Flush when: size >= N  OR  time elapsed >= T ms
Pros: bounded latency + good GPU utilization
Cons: slightly more complex implementation
This is what EAIGLE uses: size=8, timeout=50ms
```

**Strategy D — Adaptive batching (future enhancement):**
```
Measure GPU utilization → if < 60%, increase batch size
                        → if > 90%, decrease timeout
Pros: self-tuning
Cons: complex, risk of instability
```

---

### 10.4 Avoiding Memory Leaks

Three categories of potential memory leaks and their mitigations:

**1. Shared memory blocks (highest risk):**

Risk: if a consumer process crashes after reading from Redis but before `shm.unlink()`, the block is orphaned in `/dev/shm` forever.

Mitigation:
```python
# Background task runs every 10 seconds
async def _shm_cleanup_loop():
    while True:
        cleaned = cleanup_stale_shm(prefix="eaigle_", max_age_s=10.0)
        if cleaned:
            logger.info("Cleaned %d stale shm blocks", cleaned)
        await asyncio.sleep(10)
```

**2. Redis stream growth:**

Risk: if consumers stop processing (e.g., inference service down), stream depths grow until Redis hits `maxmemory`.

Mitigation:
```python
# Approximate trimming on every write
await redis.xadd(stream, message, maxlen=500, approximate=True)
```
Also: `maxmemory-policy allkeys-lru` evicts old entries under pressure.

**3. Python object accumulation:**

Risk: `FrameAccumulator` in `ResultCollector` holds Stage A results waiting for Stage B. If Stage B never arrives, the accumulator grows.

Mitigation:
```python
# Cleanup loop in result_collector.py
async def _cleanup_loop(self):
    while True:
        now = time.time()
        stale = [fid for fid, acc in self._accumulators.items()
                 if now - acc.created_at > FRAME_TIMEOUT_S]  # 2.0s
        for fid in stale:
            del self._accumulators[fid]
        await asyncio.sleep(1.0)
```

**4. ProcessPoolExecutor workers:**

Risk: worker processes accumulate memory from numpy arrays if they're not GC'd.

Mitigation:
- `_process_frame` returns a copy of the numpy array and the original shm array goes out of scope
- Python's reference counting handles deallocation immediately
- Worker processes are restarted if they exceed memory limits (configurable via OS `ulimit`)

---

### 10.5 Monitoring Strategy

**Three-tier monitoring:**

**Tier 1 — Application metrics (Prometheus + Grafana):**

Recommended Grafana dashboard panels:

| Panel | Query | Alert Threshold |
|---|---|---|
| Cameras connected | `eaigle_active_cameras` | < 45 (out of 50) |
| Frame ingestion rate | `rate(eaigle_frames_ingested_total[1m])` | < 400 FPS |
| Frame drop rate | `rate(eaigle_frames_dropped_total[1m])` | > 10 FPS |
| Preprocessing P95 latency | `histogram_quantile(0.95, eaigle_preprocessing_latency_ms_bucket)` | > 50ms |
| Inference P95 latency | `histogram_quantile(0.95, eaigle_inference_latency_ms_bucket)` | > 200ms |
| End-to-end P95 latency | `histogram_quantile(0.95, eaigle_pipeline_latency_ms_bucket)` | > 500ms |
| Detection rate | `rate(eaigle_detections_total[1m])` | Alert if 0 for > 60s |
| SHM blocks active | `eaigle_shm_blocks_active` | > 500 (memory leak) |

**Tier 2 — Infrastructure metrics (node_exporter / cloud monitoring):**

| Metric | Alert |
|---|---|
| CPU utilization (pipeline pods) | > 80% sustained |
| GPU utilization (inference pods) | > 90% sustained |
| GPU memory | > 85% |
| `/dev/shm` usage | > 1.5GB per pod |
| Redis memory | > 80% of maxmemory |
| Redis latency | P99 > 5ms |

**Tier 3 — Structured logs (ELK / CloudWatch):**

Every log line includes:
- `camera_id` — correlate issues to specific cameras
- `frame_id` — trace a frame end-to-end across all stages
- `batch_id` — correlate inference requests to results
- `latency_ms` — identify slow operations
- `pipeline_stage` — know exactly where a frame is

Example log search queries:
```
# Find all errors for camera cam_03
camera_id:cam_03 AND level:ERROR

# Trace a specific frame through all stages
frame_id:"abc-123-..."

# Find frames with latency > 200ms
pipeline_latency_ms:>200
```

---

## 11. Bonus Features Implemented

### 11.1 Asynchronous Pipeline

The entire pipeline is built on `asyncio` — **fully implemented**.

- All I/O operations (Redis, HTTP, camera reads) are non-blocking
- 50 cameras run concurrently in a single event loop with `asyncio.gather()`
- `uvloop` replaces the default event loop for 2–4× higher I/O throughput:
  ```python
  # app.py
  import uvloop
  asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
  asyncio.run(main())
  ```
- Blocking operations (cv2, file I/O) are explicitly offloaded to executor pools

### 11.2 Message Broker — Redis Streams

Redis Streams are used as the message broker — **fully implemented**.

Features used:
- `XADD` — produce messages with auto-generated IDs
- `XREADGROUP` — consume with consumer groups (load distribution)
- `XACK` — acknowledge successful processing (at-least-once guarantee)
- `XAUTOCLAIM` — recover stale messages from crashed consumers
- `PUBLISH/SUBSCRIBE` — fan-out of final hypothesis events
- `HSET` with TTL — store hypothesis results with expiry
- Stream trimming — prevent unbounded growth

Why Redis over Kafka at this scale:
- No ZooKeeper / KRaft dependency
- Lower operational overhead
- Built-in data structures (HSET, PubSub) beyond just queuing
- Sub-millisecond latency vs Kafka's 1–5ms
- Kafka preferred if scale > 1M messages/sec or multi-datacenter replication needed

### 11.3 Docker Compose Deployment

Fully implemented in `docker/docker-compose.yml`.

```bash
# Standard deployment (OpenCV backend)
cd docker && docker compose up

# With NVIDIA GPU (DeepStream backend)
docker compose --profile deepstream up

# Scale inference replicas
docker compose up --scale inference=4
```

Services: Redis, Inference (YOLOv8), Pipeline, Simulator, Prometheus.
All services share a `eaigle_net` bridge network.
Prometheus scrapes pipeline metrics automatically.

---

## 12. Future Work

### 12.1 Inference & Models

| Direction | Description |
|---|---|
| **Multi-model ensemble** | Run YOLOv8n + YOLOv8m in parallel, fuse results with weighted NMS for better accuracy/speed tradeoff |
| **Real OCR** | Replace the stub secondary stage with EasyOCR or PaddleOCR for actual license plate and badge reading |
| **Action recognition** | Add a temporal model (SlowFast, VideoMAE) on person crops to detect loitering, running, or fighting |
| **Re-identification (Re-ID)** | Track the same person/vehicle across multiple cameras using OSNet or similar embedding model |

### 12.2 Pipeline & Infrastructure

| Direction | Description |
|---|---|
| **NVIDIA Triton Inference Server** | Batch requests from all 50 cameras to a shared GPU pool with dynamic batching and model versioning |
| **GPU-accelerated preprocessing** | Replace OpenCV CPU ops with CUDA-based ops (cv2.cuda or cuCIM) to offload the ProcessPoolExecutor bottleneck |
| **Frame deduplication** | Perceptual hashing (pHash) to skip near-identical frames and reduce inference load by 30–60% on static scenes |
| **Adaptive frame rate** | Reduce polling rate per camera when scene is idle, increase when motion is detected |

### 12.3 Data & Storage

| Direction | Description |
|---|---|
| **Event-driven clip recording** | When a detection occurs, write a short video clip to object storage (MinIO/S3) with detection metadata |
| **TimescaleDB / ClickHouse** | Store detection history for analytics queries — vehicle count by hour, dwell time heatmaps |
| **Feature vector store** | Persist Re-ID embeddings in Milvus/Qdrant for cross-session identity matching |

### 12.4 Reliability

| Direction | Description |
|---|---|
| **Dead-letter queue** | Route failed inference frames to a separate Redis stream for retry or manual review |
| **Circuit breaker** | If inference service latency spikes, automatically fall back to stub and alert operators |
| **Camera health monitoring** | Detect stream disconnects, frozen frames, and blur/occlusion automatically |

### 12.5 Observability

| Direction | Description |
|---|---|
| **Grafana dashboards** | Wire existing Prometheus metrics to dashboards showing per-camera FPS, inference latency p95, detection rates |
| **OpenTelemetry tracing** | Trace a single frame end-to-end across all services with distributed spans |

### 12.6 Security

| Direction | Description |
|---|---|
| **RBAC on the API** | JWT-based access control so only authorized services can post to `/detect/*` |
| **Encrypted shared memory** | For deployments where frame data is sensitive (e.g., government or critical infrastructure sites) |

### 12.7 Recommended Priorities

The highest-ROI next steps based on the current prototype:

1. **Real OCR** — completes the secondary pipeline with actual value output
2. **NVIDIA Triton** — unlocks multi-model GPU batching and model versioning at scale
3. **TimescaleDB** — makes detections queryable for reporting and business intelligence
4. **Re-ID across cameras** — the key capability that elevates this from a detector to a tracking system

---

*Generated for EAIGLE AI project by Zhila Bahrami — 2026-03-04*
