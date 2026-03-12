"""
DynamicBatcher — assembles frames into batches using a dual trigger:
  1. Size trigger  : flush when batch reaches max_size.
  2. Time trigger  : flush when max_wait_ms has elapsed since first item.

This bounds worst-case latency while maximising GPU utilisation.
Thread/coroutine safe via asyncio.Lock.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, List

from eaigle.models.frame import FrameMetadata


class DynamicBatcher:
    def __init__(
        self,
        max_size: int,
        max_wait_ms: float,
        flush_callback: Callable[[List[FrameMetadata]], Awaitable[None]],
    ):
        self._max_size = max_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._flush_cb = flush_callback
        self._batch: List[FrameMetadata] = []
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task | None = None

    async def add(self, meta: FrameMetadata) -> None:
        async with self._lock:
            if not self._batch:
                # Start timeout timer on first item
                self._timer_task = asyncio.create_task(self._timeout_flush())
            self._batch.append(meta)

            if len(self._batch) >= self._max_size:
                if self._timer_task:
                    self._timer_task.cancel()
                    self._timer_task = None
                await self._do_flush()

    async def _timeout_flush(self) -> None:
        await asyncio.sleep(self._max_wait_s)
        async with self._lock:
            if self._batch:
                await self._do_flush()

    async def _do_flush(self) -> None:
        batch = self._batch[:]
        self._batch.clear()
        self._timer_task = None
        await self._flush_cb(batch)

    async def flush_remaining(self) -> None:
        """Drain any partially-assembled batch at shutdown."""
        async with self._lock:
            if self._timer_task:
                self._timer_task.cancel()
            if self._batch:
                await self._do_flush()
