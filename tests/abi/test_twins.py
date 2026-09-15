"""Every ctypes twin is laid out as its C++ struct: the sizes and offsets
the headers static_assert, pinned here so a drifted field fails a test."""

import ctypes

import pytest

from peregrinepy import abi

twins = {
    # struct, size in bytes, (field, offset) pairs
    abi.pgView: (72, (("data", 0), ("rank", 8), ("extent", 12), ("stride", 32))),
    abi.pgCells: (32, (("start", 0), ("extent", 12), ("n", 28))),
    abi.pgDims: (12, (("ni", 0), ("nj", 4), ("nk", 8))),
    abi.pgTiling: (40, (("cells", 24), ("count", 32), ("tiles", 36))),
    # the member twins, as arrays.hpp asserts them
    abi.Column: (32, (("records", 0), ("at", 8), ("entry", 16))),
    abi.FaceColumn: (40, (("records", 0), ("at", 8), ("nface", 32))),
    abi.PerEntryInt: (16, (("all", 0), ("value", 8))),
    abi.Record: (72, (("r", 0),)),
    abi.DimsColumn: (16, (("all", 0), ("at", 8))),
}


@pytest.mark.parametrize("twin", list(twins), ids=lambda t: t.__name__)
def test_twin(twin):
    size, offsets = twins[twin]
    assert ctypes.sizeof(twin) == size
    for field, offset in offsets:
        assert getattr(twin, field).offset == offset
