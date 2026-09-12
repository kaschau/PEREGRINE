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


lib.pgKineticTheory.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    ctypes.c_double,
    _range,
]


def kineticTheory(blk, th, nface):
    lib.pgKineticTheory(
        _rec(blk.q),
        _rec(blk.qt),
        _rec(th.MW),
        _rec(th.dij),
        _rec(th.kappaPoly),
        _rec(th.muPoly),
        th.Ru,
        ctypes.byref(cellRange(blk, nface)),
    )


lib.pgKineticTheoryUnityLewis.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    ctypes.c_double,
    _range,
]


def kineticTheoryUnityLewis(blk, th, nface):
    lib.pgKineticTheoryUnityLewis(
        _rec(blk.Q),
        _rec(blk.q),
        _rec(blk.qh),
        _rec(blk.qt),
        _rec(th.MW),
        _rec(th.kappaPoly),
        _rec(th.lewis),
        _rec(th.muPoly),
        th.Ru,
        ctypes.byref(cellRange(blk, nface)),
    )


lib.pgChungDenseGasUnityLewis.argtypes = [
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
    _view,
    _view,
    ctypes.c_double,
    _range,
]


def chungDenseGasUnityLewis(blk, th, nface):
    lib.pgChungDenseGasUnityLewis(
        _rec(blk.Q),
        _rec(blk.q),
        _rec(blk.qh),
        _rec(blk.qt),
        _rec(th.MW),
        _rec(th.Tcrit),
        _rec(th.Vcrit),
        _rec(th.acentric),
        _rec(th.chungA),
        _rec(th.chungB),
        _rec(th.lewis),
        _rec(th.redDipole),
        th.Ru,
        ctypes.byref(cellRange(blk, nface)),
    )


lib.pgConstantProps.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    ctypes.c_double,
    _range,
]


def constantProps(blk, th, nface):
    lib.pgConstantProps(
        _rec(blk.Q),
        _rec(blk.q),
        _rec(blk.qh),
        _rec(blk.qt),
        _rec(th.MW),
        _rec(th.kappa0),
        _rec(th.lewis),
        _rec(th.mu0),
        th.Ru,
        ctypes.byref(cellRange(blk, nface)),
    )
