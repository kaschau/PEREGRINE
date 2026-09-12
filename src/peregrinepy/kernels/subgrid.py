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


lib.pgSmagorinsky.argtypes = [_view, _view, _view, _view, _view, _dims]


def smagorinsky(blk):
    lib.pgSmagorinsky(
        _rec(blk.J),
        _rec(blk.Q),
        _rec(blk.grads),
        _rec(blk.qh),
        _rec(blk.qt),
        _dimsOf(blk),
    )
