

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import yaml
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

from eaigle.aggregation.hypothesis_fusion import HypothesisFusion
from eaigle.aggregation.hypothesis_store import HypothesisStore
from eaigle.aggregation.result_collector import ResultCollector
from eaigle.ingestion.camera_manager import CameraStreamManager
from eaigle.inference.dispatcher import InferenceDispatcher
from eaigle.models.hypothesis import CameraConfig
from eaigle.observability.logging_config import configure_logging
from eaigle.observability.metrics import start_metrics_server
from eaigle.preprocessing.shm_utils import cleanup_stale_shm
from eaigle.preprocessing.worker_pool import PreprocessingWorkerPool
from eaigle.transport.redis_client import RedisClient
logger = logging.getLogger(__name__)

def _build_ingestion_backend(
    backend: str,
    cam_cfgs: list,
    redis: "RedisClient",
    stop_event: asyncio.Event,
    ds_cfg: dict,
):
    
    if backend == "deepstream":
        try:
            from eaigle.ingestion.deepstream_manager import DeepStreamIngestionManager
            mgr = DeepStreamIngestionManager(
                camera_configs=cam_cfgs,
                redis=redis,
                stop_event=stop_event,
                output_width=ds_cfg.get("output_width", 1280),
                output_height=ds_cfg.get("output_height", 720),
            )
            logger.info("Ingestion backend: NVIDIA DeepStream")
            return mgr
        except ImportError as exc:
            logger.warning(
                "DeepStream unavailable (%s) — falling back to OpenCV backend", exc
            )

    logger.info("Ingestion backend: OpenCV + FFmpeg")
    return CameraStreamManager(
        camera_configs=cam_cfgs,
        redis=redis,
        stop_event=stop_event,
    )

async def _shm_cleanup_loop(stop_event: asyncio.Event) -> None:
    
    while not stop_event.is_set():
        await asyncio.sleep(10)
        n = cleanup_stale_shm(max_age_s=15.0)
        if n:
            logger.debug("Cleaned up %d stale shm blocks", n)

async def main(cfg: dict) -> None:
    obs = cfg.get("observability", {})
    configure_logging(
        level=obs.get("log_level", "INFO"),
        json_logs=obs.get("json_logs", False),
    )
    start_metrics_server(port=obs.get("metrics_port", 9090))
    logger.info("EAIGLE AI pipeline starting")

    redis_cfg = cfg.get("redis", {})
    redis = await RedisClient.create(redis_cfg.get("url", "redis://localhost:6379/0"))
    logger.info("Redis connected")

    cam_cfgs = [
        CameraConfig.from_dict(c) for c in cfg["cameras"]["streams"]
    ]
    camera_ids = [c.camera_id for c in cam_cfgs]
    camera_zones = {c.camera_id: c.zone for c in cam_cfgs}

    stop_event = asyncio.Event()

    def _handle_signal(*_) -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    inf_cfg     = cfg.get("inference", {})
    agg_cfg     = cfg.get("aggregation", {})
    pp_cfg      = cfg.get("preprocessing", {})
    ingestion   = cfg.get("ingestion", {})

    camera_mgr = _build_ingestion_backend(
        backend=ingestion.get("backend", "opencv"),
        cam_cfgs=cam_cfgs,
        redis=redis,
        stop_event=stop_event,
        ds_cfg=ingestion.get("deepstream", {}),
    )

    pp_pool = PreprocessingWorkerPool(
        redis=redis,
        camera_ids=camera_ids,
        pipeline_cfg=pp_cfg.get("pipeline", {}),
        num_workers=pp_cfg.get("num_workers", 4),
    )

    dispatcher = InferenceDispatcher(
        redis=redis,
        inference_url=inf_cfg.get("url", "http://localhost:8001"),
        batch_size=inf_cfg.get("batch_size", 8),
        batch_timeout_ms=inf_cfg.get("batch_timeout_ms", 50.0),
        http_timeout_s=inf_cfg.get("http_timeout_s", 5.0),
    )

    fusion = HypothesisFusion(
        nms_iou_threshold=agg_cfg.get("nms_iou_threshold", 0.45),
        min_confidence=agg_cfg.get("min_confidence", 0.40),
        temporal_window=agg_cfg.get("temporal_window", 5),
        temporal_min_presence=agg_cfg.get("temporal_min_presence", 2),
        camera_zones=camera_zones,
    )

    store = HypothesisStore(redis=redis)

    collector = ResultCollector(
        redis=redis,
        fusion=fusion,
        store=store,
    )

    logger.info(
        "Starting pipeline: %d cameras, %d pp workers, batch_size=%d",
        len(cam_cfgs), pp_cfg.get("num_workers", 4), inf_cfg.get("batch_size", 8),
    )

    await asyncio.gather(
        camera_mgr.run(),
        pp_pool.run(stop_event),
        dispatcher.run(stop_event),
        collector.run(stop_event),
        _shm_cleanup_loop(stop_event),
        return_exceptions=True,
    )

    await redis.close()
    logger.info("Pipeline shut down cleanly")

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="EAIGLE AI pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML config",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    asyncio.run(main(config))
