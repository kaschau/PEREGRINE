"""Generated from the kernels' bodies: each takes the block and whatever
else it reads, and calls the C function."""

import ctypes

import numpy as np

from ..abi import Dims, Range, View, cellRange, lib

_view, _dims, _range = ctypes.POINTER(View), ctypes.POINTER(Dims), ctypes.POINTER(Range)


def _rec(array):
    return ctypes.byref(View.of(array))


def _dimsOf(blk):
    return ctypes.byref(Dims(blk.ni, blk.nj, blk.nk, blk.ng))


def _doubles(values):
    return np.ascontiguousarray(values, dtype=np.float64).ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )


lib.pgJamesonPressure.argtypes = [_view, _view, _dims]


def jamesonPressure(blk):
    lib.pgJamesonPressure(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))


lib.pgVanAlbadaPressure.argtypes = [_view, _view, _dims]


def vanAlbadaPressure(blk):
    lib.pgVanAlbadaPressure(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))


lib.pgVanLeer.argtypes = [_view, _view, _dims]


def vanLeer(blk):
    lib.pgVanLeer(_rec(blk.phi), _rec(blk.q), _dimsOf(blk))
