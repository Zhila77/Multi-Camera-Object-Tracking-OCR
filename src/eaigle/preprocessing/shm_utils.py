
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Tuple

import numpy as np

def write_frame_to_shm(frame_id: str, frame: np.ndarray) -> str:
    
    shm_name = f"eaigle_{frame_id}"
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=frame.nbytes)
    buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    buf[:] = frame[:]
    shm.close()
    return shm_name

def read_frame_from_shm(
    shm_name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype | str,
) -> np.ndarray:
    
    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    result = frame.copy()
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    return result

def cleanup_stale_shm(prefix: str = "eaigle_", max_age_s: float = 10.0) -> int:
    
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
