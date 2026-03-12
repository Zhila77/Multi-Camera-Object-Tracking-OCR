"""
POSIX Shared Memory helpers.

Pixel data is large (1920×1080 BGR ≈ 6 MB).  Passing it through Redis on
every hop would saturate memory bandwidth.  Instead:
  - Writer: allocates a named /dev/shm block and copies frame bytes in.
  - Reader: maps the block, copies frame out, then unlinks (frees) the block.

This gives zero-serialisation transfer between pipeline stages on the same host.
For multi-host deployments the design can be swapped to object-store references.
"""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Tuple

import numpy as np


def write_frame_to_shm(frame_id: str, frame: np.ndarray) -> str:
    """
    Write *frame* into a new POSIX shared memory block.
    Returns the shm name so consumers can map it later.
    The block persists until the consumer calls shm.unlink().
    """
    shm_name = f"eaigle_{frame_id}"
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=frame.nbytes)
    buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    buf[:] = frame[:]
    shm.close()          # Close our handle; data stays alive until unlink
    return shm_name


def read_frame_from_shm(
    shm_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype | str,
) -> np.ndarray:
    """
    Read a frame from shared memory, return a copy, then unlink the block.
    After this call the shm block is freed.
    """
    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    result = frame.copy()
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass   # Already cleaned up by another worker (OK)
    return result


def cleanup_stale_shm(prefix: str = "eaigle_", max_age_s: float = 10.0) -> int:
    """
    Scan /dev/shm for blocks with the given prefix that are older than
    max_age_s seconds and unlink them.  Returns number of blocks cleaned.
    Meant to run in a background asyncio task to prevent shm exhaustion.
    """
    import os, time

    cleaned = 0
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return 0

    now = time.time()
    for name in os.listdir(shm_dir):
        if not name.startswith(prefix):
            continue
        path = os.path.join(shm_dir, name)
        try:
            age = now - os.stat(path).st_mtime
            if age > max_age_s:
                os.unlink(path)
                cleaned += 1
        except OSError:
            pass
    return cleaned
