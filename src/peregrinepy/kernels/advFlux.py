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


lib.pgKEEP.argtypes = [
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
]


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


lib.pgKEEPpe.argtypes = [
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
]


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


lib.pgKEPaEC.argtypes = [
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
]


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


lib.pgAusmPlusUp.argtypes = [
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
]


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


lib.pgCentralDifference.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _dims,
]


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


lib.pgFourthOrderKEEP.argtypes = [
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
]


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


lib.pgHllc.argtypes = [
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
]


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


lib.pgMuscl2hllc.argtypes = [
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
]


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


lib.pgMuscl2rusanov.argtypes = [
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
]


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


lib.pgMyKEEP.argtypes = [
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
]


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


lib.pgRusanov.argtypes = [
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
]


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


lib.pgScalarDissipation.argtypes = [
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
]


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
