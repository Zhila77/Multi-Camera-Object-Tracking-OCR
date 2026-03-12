"""
DeepStreamIngestionManager — NVIDIA DeepStream-based multi-camera RTSP ingestion.

Architecture
------------
GStreamer pipeline (runs in a GLib main-loop thread):

  nvurisrcbin(rtsp://cam_01) ──┐
  nvurisrcbin(rtsp://cam_02) ──┼──► nvstreammux ──► [pad probe] ──► fakesink
  nvurisrcbin(rtsp://cam_N)  ──┘        │
                                         └─ batches N frames per tick

Pad probe extracts each frame from the GPU-memory batch via
pyds.get_nvds_buf_surface(), writes pixel data to POSIX shared memory,
and schedules a Redis Stream publish on the asyncio event loop via
asyncio.run_coroutine_threadsafe().

The rest of the pipeline (preprocessing, inference, aggregation) is
unchanged — DeepStream is a drop-in replacement for the OpenCV backend.

Advantages over OpenCV backend
-------------------------------
- NVDEC hardware H.264/H.265 decode — frees CPU entirely for preprocessing
- nvstreammux native batching — frames from all N cameras arrive in one
  GPU buffer, aligned in time
- nvurisrcbin handles RTSP reconnection internally (no custom backoff code)
- GPU zero-copy until get_nvds_buf_surface() maps to CPU — one copy total
  vs. OpenCV's two copies (decode → cv2 buffer → numpy)

Requirements
------------
- NVIDIA GPU + CUDA ≥ 11.0
- NVIDIA DeepStream SDK ≥ 6.x  (provides GStreamer plugins + pyds bindings)
- GStreamer 1.x  (gi.repository.Gst)
- pyds  (pip install pyds)
- Python ≥ 3.11

Fallback
--------
If DeepStream is not installed, importing this module raises ImportError.
app.py catches this and falls back to the OpenCV CameraStreamManager.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional

# ------------------------------------------------------------------
# Optional imports — DeepStream / GStreamer
# ------------------------------------------------------------------
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # type: ignore
    import pyds  # type: ignore
    Gst.init(None)
    DEEPSTREAM_AVAILABLE = True
except Exception:
    DEEPSTREAM_AVAILABLE = False

import numpy as np

from eaigle.models.frame import FrameMetadata
from eaigle.models.hypothesis import CameraConfig
from eaigle.preprocessing.shm_utils import write_frame_to_shm
from eaigle.transport.redis_client import RedisClient
from eaigle.transport.stream_producer import StreamProducer

logger = logging.getLogger(__name__)

# nvstreammux: maximum time (µs) to wait before flushing a partial batch
_BATCHED_PUSH_TIMEOUT_US = 40_000   # 40 ms → guarantees ≤40ms added latency
_GPU_ID = 0


class DeepStreamIngestionManager:
    """
    Manages all RTSP camera streams via NVIDIA DeepStream.

    Lifecycle:
        mgr = DeepStreamIngestionManager(configs, redis, stop_event)
        await mgr.run()   # blocks until stop_event is set
    """

    def __init__(
        self,
        camera_configs: List[CameraConfig],
        redis: RedisClient,
        stop_event: asyncio.Event,
        output_width: int = 1280,
        output_height: int = 720,
    ):
        if not DEEPSTREAM_AVAILABLE:
            raise ImportError(
                "NVIDIA DeepStream is not installed.\n"
                "Install: NVIDIA DeepStream SDK ≥ 6.x and 'pip install pyds'.\n"
                "Set ingestion.backend: opencv in pipeline.yaml to use OpenCV instead."
            )

        self._configs = camera_configs
        self._redis = redis
        self._stop = stop_event
        self._out_w = output_width
        self._out_h = output_height
        self._producer = StreamProducer(redis)

        # Maps source_id (int) → CameraConfig
        self._source_map: Dict[int, CameraConfig] = {
            i: cfg for i, cfg in enumerate(camera_configs)
        }
        # Per-camera monotonic frame counter
        self._seq_counters: Dict[str, int] = {
            cfg.camera_id: 0 for cfg in camera_configs
        }

        self._pipeline: Optional["Gst.Pipeline"] = None
        self._glib_loop: Optional["GLib.MainLoop"] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._asyncio_loop = asyncio.get_running_loop()

        # Ensure Redis consumer groups exist before any frames arrive
        for cfg in self._configs:
            await self._producer.ensure_group(
                f"raw_frames:{cfg.camera_id}", "preprocessing_workers"
            )

        # Run GLib main loop in a daemon thread so asyncio remains unblocked
        gst_thread = threading.Thread(
            target=self._run_gstreamer_blocking,
            name="deepstream_glib",
            daemon=True,
        )
        gst_thread.start()
        logger.info(
            "DeepStream pipeline started (%d camera(s))", len(self._configs)
        )

        # Await stop_event in asyncio; then signal GLib to quit
        await self._stop.wait()
        self._shutdown_gstreamer()

        # Give GLib thread a moment to drain
        gst_thread.join(timeout=5.0)
        logger.info("DeepStream pipeline shut down")

    # ------------------------------------------------------------------
    # GStreamer pipeline construction
    # ------------------------------------------------------------------

    def _run_gstreamer_blocking(self) -> None:
        """Runs in the GLib thread. Blocks until quit() is called."""
        self._pipeline = self._build_pipeline()
        self._glib_loop = GLib.MainLoop()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to set DeepStream pipeline to PLAYING")
            return

        logger.info("GLib main loop running")
        self._glib_loop.run()

        self._pipeline.set_state(Gst.State.NULL)
        logger.info("GLib main loop exited, pipeline set to NULL")

    def _build_pipeline(self) -> "Gst.Pipeline":
        pipeline = Gst.Pipeline.new("eaigle_ds_pipeline")

        # ---- nvstreammux ----
        mux = self._make_element(pipeline, "nvstreammux", "mux")
        mux.set_property("batch-size", len(self._configs))
        mux.set_property("width", self._out_w)
        mux.set_property("height", self._out_h)
        mux.set_property("batched-push-timeout", _BATCHED_PUSH_TIMEOUT_US)
        mux.set_property("live-source", True)
        mux.set_property("gpu-id", _GPU_ID)

        # ---- One nvurisrcbin per camera ----
        for source_id, cfg in self._source_map.items():
            src = self._make_element(pipeline, "nvurisrcbin", f"src_{source_id}")
            src.set_property("uri", cfg.rtsp_url)
            src.set_property("source-id", source_id)
            src.set_property("gpu-id", _GPU_ID)
            # DeepStream 6+ supports rtsp-reconnect-interval-ms
            src.set_property("rtsp-reconnect-interval", 5)

            # nvurisrcbin has dynamic pads; link when pad-added fires
            src.connect("pad-added", self._on_pad_added, mux, source_id)

        # ---- nvvideoconvert: GPU colour conversion ----
        # Converts from NV12 (decoder output) to RGBA so pyds can read pixels
        converter = self._make_element(pipeline, "nvvideoconvert", "converter")
        converter.set_property("gpu-id", _GPU_ID)

        # ---- capsfilter: request RGBA so get_nvds_buf_surface gives RGBA ----
        caps_filter = self._make_element(pipeline, "capsfilter", "caps")
        caps_filter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )

        # ---- fakesink: we extract data in the pad probe, not here ----
        sink = self._make_element(pipeline, "fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)

        # ---- Link: mux → converter → caps_filter → sink ----
        if not mux.link(converter):
            raise RuntimeError("Failed to link nvstreammux → nvvideoconvert")
        if not converter.link(caps_filter):
            raise RuntimeError("Failed to link nvvideoconvert → capsfilter")
        if not caps_filter.link(sink):
            raise RuntimeError("Failed to link capsfilter → fakesink")

        # ---- Pad probe on mux src pad ----
        # This fires once per batch (all cameras, one GStreamer buffer)
        mux_src_pad = mux.get_static_pad("src")
        mux_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._on_batch_buffer,
            None,
        )

        return pipeline

    @staticmethod
    def _make_element(
        pipeline: "Gst.Pipeline", factory: str, name: str
    ) -> "Gst.Element":
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(
                f"Could not create GStreamer element '{factory}'. "
                f"Ensure DeepStream plugins are installed "
                f"(gst-inspect-1.0 {factory})."
            )
        pipeline.add(el)
        return el

    # ------------------------------------------------------------------
    # Dynamic pad linking (nvurisrcbin → nvstreammux)
    # ------------------------------------------------------------------

    def _on_pad_added(
        self,
        src_element: "Gst.Element",
        new_pad: "Gst.Pad",
        mux: "Gst.Element",
        source_id: int,
    ) -> None:
        """
        Called when nvurisrcbin exposes its src pad.
        Links it to the corresponding nvstreammux sink pad.
        """
        sink_pad_name = f"sink_{source_id}"
        sink_pad = mux.get_request_pad(sink_pad_name)
        if sink_pad is None:
            logger.error(
                "Could not get mux sink pad %s for source %d",
                sink_pad_name, source_id,
            )
            return

        if sink_pad.is_linked():
            logger.debug("Mux sink pad %s already linked", sink_pad_name)
            return

        link_ret = new_pad.link(sink_pad)
        if link_ret != Gst.PadLinkReturn.OK:
            logger.error(
                "Failed to link source %d → mux %s: %s",
                source_id, sink_pad_name, link_ret,
            )
        else:
            cam_id = self._source_map[source_id].camera_id
            logger.info("Linked camera %s (source %d) to mux", cam_id, source_id)

    # ------------------------------------------------------------------
    # Pad probe — frame extraction
    # ------------------------------------------------------------------

    def _on_batch_buffer(
        self,
        pad: "Gst.Pad",
        info: "Gst.PadProbeInfo",
        user_data: object,
    ) -> "Gst.PadProbeReturn":
        """
        Called by GStreamer in its streaming thread for every batched buffer.

        Extracts per-frame numpy arrays using pyds, writes each to POSIX
        shared memory, then schedules a Redis publish on the asyncio loop.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        # Retrieve the NvDsBatchMeta attached to this GStreamer buffer
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            batch_id  = frame_meta.batch_id
            frame_num = frame_meta.frame_num

            cfg = self._source_map.get(source_id)
            if cfg is None:
                logger.warning("Unknown source_id %d in batch", source_id)
                try:
                    frame_list = frame_list.next
                except StopIteration:
                    break
                continue

            try:
                # get_nvds_buf_surface maps the NvBufSurface to CPU and
                # returns a numpy view (H, W, 4) in RGBA order.
                # .copy() materialises a real numpy array before the buffer
                # is released back to the GStreamer pool.
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
                # Drop alpha channel: RGBA → RGB
                frame_rgb = n_frame[:, :, :3].copy()
            except Exception as exc:
                logger.error(
                    "get_nvds_buf_surface failed for cam %s: %s",
                    cfg.camera_id, exc,
                )
                try:
                    frame_list = frame_list.next
                except StopIteration:
                    break
                continue

            self._publish_frame(cfg.camera_id, frame_num, frame_rgb)

            try:
                frame_list = frame_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Frame → shm → Redis (bridge GLib thread → asyncio loop)
    # ------------------------------------------------------------------

    def _publish_frame(
        self, camera_id: str, frame_num: int, frame: np.ndarray
    ) -> None:
        """
        Writes pixel data to shared memory, then schedules a coroutine
        on the asyncio event loop from the GLib streaming thread.
        """
        frame_id = str(uuid.uuid4())
        try:
            shm_key = write_frame_to_shm(frame_id, frame)
        except Exception as exc:
            logger.error("shm write failed for %s frame %d: %s",
                         camera_id, frame_num, exc)
            return

        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1
        seq = self._seq_counters[camera_id]
        self._seq_counters[camera_id] = seq + 1

        meta = FrameMetadata(
            frame_id=frame_id,
            camera_id=camera_id,
            capture_ts=time.time(),
            sequence_num=seq,
            shm_key=shm_key,
            width=w,
            height=h,
            channels=c,
            dtype=str(frame.dtype),
            pipeline_stage="raw",
        )

        # Thread-safe: schedule the coroutine on the asyncio event loop
        if self._asyncio_loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._producer.publish(
                    stream_name=f"raw_frames:{camera_id}",
                    message=meta.to_dict(),
                ),
                self._asyncio_loop,
            )

    # ------------------------------------------------------------------
    # GStreamer bus messages
    # ------------------------------------------------------------------

    def _on_bus_message(
        self, bus: "Gst.Bus", message: "Gst.Message"
    ) -> None:
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("DeepStream: EOS received")
            if self._glib_loop:
                self._glib_loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("DeepStream GStreamer error: %s\nDebug: %s", err, debug)
            if self._glib_loop:
                self._glib_loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning("DeepStream warning: %s\nDebug: %s", warn, debug)
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                _, new_state, _ = message.parse_state_changed()
                logger.debug(
                    "Pipeline state → %s", Gst.Element.state_get_name(new_state)
                )

    def _shutdown_gstreamer(self) -> None:
        """Send EOS into the pipeline and quit the GLib loop."""
        if self._pipeline:
            logger.info("Sending EOS to DeepStream pipeline")
            self._pipeline.send_event(Gst.Event.new_eos())
        if self._glib_loop and self._glib_loop.is_running():
            # Give EOS a moment to propagate, then force quit
            def _force_quit():
                if self._glib_loop and self._glib_loop.is_running():
                    self._glib_loop.quit()
            GLib.timeout_add(1500, _force_quit)
