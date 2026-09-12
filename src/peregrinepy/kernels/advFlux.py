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


lib.declare(
    "pgKEEP",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def KEEP(blk):
    lib.pgKEEP(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgKEEPpe",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def KEEPpe(blk):
    lib.pgKEEPpe(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgKEPaEC",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def KEPaEC(blk):
    lib.pgKEPaEC(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgAusmPlusUp",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def ausmPlusUp(blk):
    lib.pgAusmPlusUp(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgCentralDifference",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def centralDifference(blk):
    lib.pgCentralDifference(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _dimsOf(blk),
    )


lib.declare(
    "pgFourthOrderKEEP",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def fourthOrderKEEP(blk):
    lib.pgFourthOrderKEEP(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgHllc",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def hllc(blk):
    lib.pgHllc(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgMuscl2hllc",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def muscl2hllc(blk):
    lib.pgMuscl2hllc(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgMuscl2rusanov",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def muscl2rusanov(blk):
    lib.pgMuscl2rusanov(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgMyKEEP",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def myKEEP(blk):
    lib.pgMyKEEP(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgRusanov",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def rusanov(blk):
    lib.pgRusanov(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )


lib.declare(
    "pgScalarDissipation",
    [
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _view,
        _dims,
    ],
)


def scalarDissipation(blk):
    lib.pgScalarDissipation(
        _rec(blk.Q),
        _rec(blk.iF),
        _rec(blk.iS),
        _rec(blk.jF),
        _rec(blk.jS),
        _rec(blk.kF),
        _rec(blk.kS),
        _rec(blk.phi),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
    )
