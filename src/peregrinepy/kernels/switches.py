"""Generated from the kernels' bodies: each takes the block and whatever
else it reads, and calls the C function."""

import ctypes

import numpy as np

from ..abi import Dims, Range, View, lib

_view, _dims, _range = ctypes.POINTER(View), ctypes.POINTER(Dims), ctypes.POINTER(Range)


def _rec(array):
    return ctypes.byref(View.of(array))


def _dimsOf(blk):
    return ctypes.byref(Dims.of(blk))


def _doubles(values):
    return np.ascontiguousarray(values, dtype=np.float64).ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )


lib.declare("pgJamesonPressure", [_view, _view, _dims])


def jamesonPressure(blk):
    lib.pgJamesonPressure(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))


lib.declare("pgVanAlbadaPressure", [_view, _view, _dims])


def vanAlbadaPressure(blk):
    lib.pgVanAlbadaPressure(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))


lib.declare("pgVanLeer", [_view, _view, _dims])


def vanLeer(blk):
    lib.pgVanLeer(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))
