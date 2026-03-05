
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds
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

_BATCHED_PUSH_TIMEOUT_US = 40_000
_GPU_ID = 0

class DeepStreamIngestionManager:
    

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

        self._source_map: Dict[int, CameraConfig] = {
            i: cfg for i, cfg in enumerate(camera_configs)
        }

        self._seq_counters: Dict[str, int] = {
            cfg.camera_id: 0 for cfg in camera_configs
        }

        self._pipeline: Optional["Gst.Pipeline"] = None
        self._glib_loop: Optional["GLib.MainLoop"] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

    async def run(self) -> None:
        self._asyncio_loop = asyncio.get_running_loop()

        for cfg in self._configs:
            await self._producer.ensure_group(
                f"raw_frames:{cfg.camera_id}", "preprocessing_workers"
            )

        gst_thread = threading.Thread(
            target=self._run_gstreamer_blocking,
            name="deepstream_glib",
            daemon=True,
        )
        gst_thread.start()
        logger.info(
            "DeepStream pipeline started (%d camera(s))", len(self._configs)
        )

        await self._stop.wait()
        self._shutdown_gstreamer()

        gst_thread.join(timeout=5.0)
        logger.info("DeepStream pipeline shut down")

    def _run_gstreamer_blocking(self) -> None:
        
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

        mux = self._make_element(pipeline, "nvstreammux", "mux")
        mux.set_property("batch-size", len(self._configs))
        mux.set_property("width", self._out_w)
        mux.set_property("height", self._out_h)
        mux.set_property("batched-push-timeout", _BATCHED_PUSH_TIMEOUT_US)
        mux.set_property("live-source", True)
        mux.set_property("gpu-id", _GPU_ID)

        for source_id, cfg in self._source_map.items():
            src = self._make_element(pipeline, "nvurisrcbin", f"src_{source_id}")
            src.set_property("uri", cfg.rtsp_url)
            src.set_property("source-id", source_id)
            src.set_property("gpu-id", _GPU_ID)

            src.set_property("rtsp-reconnect-interval", 5)

            src.connect("pad-added", self._on_pad_added, mux, source_id)

        converter = self._make_element(pipeline, "nvvideoconvert", "converter")
        converter.set_property("gpu-id", _GPU_ID)

        caps_filter = self._make_element(pipeline, "capsfilter", "caps")
        caps_filter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
        )

        sink = self._make_element(pipeline, "fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)

        if not mux.link(converter):
            raise RuntimeError("Failed to link nvstreammux → nvvideoconvert")
        if not converter.link(caps_filter):
            raise RuntimeError("Failed to link nvvideoconvert → capsfilter")
        if not caps_filter.link(sink):
            raise RuntimeError("Failed to link capsfilter → fakesink")

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

    def _on_pad_added(
        self,
        src_element: "Gst.Element",
        new_pad: "Gst.Pad",
        mux: "Gst.Element",
        source_id: int,
    ) -> None:
        
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

    def _on_batch_buffer(
        self,
        pad: "Gst.Pad",
        info: "Gst.PadProbeInfo",
        user_data: object,
    ) -> "Gst.PadProbeReturn":
        
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

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

                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)

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

    def _publish_frame(
        self, camera_id: str, frame_num: int, frame: np.ndarray
    ) -> None:
        
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

        if self._asyncio_loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._producer.publish(
                    stream_name=f"raw_frames:{camera_id}",
                    message=meta.to_dict(),
                ),
                self._asyncio_loop,
            )

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
        
        if self._pipeline:
            logger.info("Sending EOS to DeepStream pipeline")
            self._pipeline.send_event(Gst.Event.new_eos())
        if self._glib_loop and self._glib_loop.is_running():

            def _force_quit():
                if self._glib_loop and self._glib_loop.is_running():
                    self._glib_loop.quit()
            GLib.timeout_add(1500, _force_quit)
