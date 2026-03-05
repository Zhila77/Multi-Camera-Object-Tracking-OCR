"""Unit tests for DynamicBatcher."""
import asyncio
import time

import pytest

from eaigle.inference.batcher import DynamicBatcher
from eaigle.models.frame import FrameMetadata


def _make_meta(camera_id: str = "cam_01", seq: int = 0) -> FrameMetadata:
    return FrameMetadata(
        frame_id=f"frame_{seq}",
        camera_id=camera_id,
        capture_ts=time.time(),
        sequence_num=seq,
        shm_key=f"shm_{seq}",
        width=640, height=480, channels=3,
        dtype="uint8",
        pipeline_stage="preprocessed",
    )


class TestDynamicBatcher:
    @pytest.mark.asyncio
    async def test_size_trigger_fires_at_max_size(self):
        flushed = []

        async def cb(batch):
            flushed.append(batch)

        batcher = DynamicBatcher(max_size=3, max_wait_ms=5000, flush_callback=cb)
        for i in range(3):
            await batcher.add(_make_meta(seq=i))

        assert len(flushed) == 1
        assert len(flushed[0]) == 3

    @pytest.mark.asyncio
    async def test_time_trigger_fires_before_max_size(self):
        flushed = []

        async def cb(batch):
            flushed.append(batch)

        batcher = DynamicBatcher(max_size=100, max_wait_ms=50, flush_callback=cb)
        await batcher.add(_make_meta(seq=0))
        await asyncio.sleep(0.12)   # wait longer than 50ms

        assert len(flushed) == 1
        assert len(flushed[0]) == 1

    @pytest.mark.asyncio
    async def test_flush_remaining_drains_partial_batch(self):
        flushed = []

        async def cb(batch):
            flushed.append(batch)

        batcher = DynamicBatcher(max_size=10, max_wait_ms=5000, flush_callback=cb)
        await batcher.add(_make_meta(seq=0))
        await batcher.add(_make_meta(seq=1))
        await batcher.flush_remaining()

        assert len(flushed) == 1
        assert len(flushed[0]) == 2

    @pytest.mark.asyncio
    async def test_multiple_size_trigger_batches(self):
        flushed = []

        async def cb(batch):
            flushed.append(batch)

        batcher = DynamicBatcher(max_size=2, max_wait_ms=5000, flush_callback=cb)
        for i in range(6):
            await batcher.add(_make_meta(seq=i))

        assert len(flushed) == 3
        assert all(len(b) == 2 for b in flushed)
