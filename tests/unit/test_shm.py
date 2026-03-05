"""Unit tests for shared memory utilities."""
import numpy as np
import pytest

from eaigle.preprocessing.shm_utils import read_frame_from_shm, write_frame_to_shm


def test_write_read_round_trip():
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    frame_id = "test_shm_round_trip"

    shm_key = write_frame_to_shm(frame_id, frame)
    assert shm_key == f"eaigle_{frame_id}"

    recovered = read_frame_from_shm(shm_key, frame.shape, frame.dtype)
    assert np.array_equal(frame, recovered)


def test_unlink_frees_block():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    shm_key = write_frame_to_shm("test_unlink", frame)
    read_frame_from_shm(shm_key, frame.shape, frame.dtype)   # also unlinks

    # After unlink, re-reading should fail
    with pytest.raises(FileNotFoundError):
        from multiprocessing import shared_memory
        shared_memory.SharedMemory(name=shm_key, create=False)
