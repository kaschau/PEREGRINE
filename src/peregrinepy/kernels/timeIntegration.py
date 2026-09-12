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


lib.pgRk4s1.argtypes = [_view, _view, _view, _view, _dims, ctypes.c_double]


def rk4s1(blk, dt):
    lib.pgRk4s1(_rec(blk.Q), _rec(blk.Q0), _rec(blk.Q1), _rec(blk.dQ), _dimsOf(blk), dt)


lib.pgRk4s2.argtypes = [_view, _view, _view, _view, _dims, ctypes.c_double]


def rk4s2(blk, dt):
    lib.pgRk4s2(_rec(blk.Q), _rec(blk.Q0), _rec(blk.Q2), _rec(blk.dQ), _dimsOf(blk), dt)


lib.pgRk4s3.argtypes = [_view, _view, _view, _view, _dims, ctypes.c_double]


def rk4s3(blk, dt):
    lib.pgRk4s3(_rec(blk.Q), _rec(blk.Q0), _rec(blk.Q3), _rec(blk.dQ), _dimsOf(blk), dt)


lib.pgRk4s4.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _dims,
    ctypes.c_double,
]


def rk4s4(blk, dt):
    lib.pgRk4s4(
        _rec(blk.Q),
        _rec(blk.Q0),
        _rec(blk.Q1),
        _rec(blk.Q2),
        _rec(blk.Q3),
        _rec(blk.dQ),
        _dimsOf(blk),
        dt,
    )


lib.pgDQdt.argtypes = [_view, _view, _view, _view, _dims, ctypes.c_double]


def dQdt(blk, dt):
    lib.pgDQdt(
        _rec(blk.Q), _rec(blk.Qn), _rec(blk.Qnm1), _rec(blk.dQ), _dimsOf(blk), dt
    )


lib.pgLocalDtau.argtypes = [
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
    ctypes.c_bool,
]


def localDtau(blk, viscous):
    lib.pgLocalDtau(
        _rec(blk.Q),
        _rec(blk.dIJK),
        _rec(blk.dtau),
        _rec(blk.iS),
        _rec(blk.jS),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _rec(blk.qt),
        _dimsOf(blk),
        viscous,
    )


lib.pgDTrk3s1.argtypes = [_view, _view, _view, _view, _dims]


def DTrk3s1(blk):
    lib.pgDTrk3s1(_rec(blk.Q0), _rec(blk.dQ), _rec(blk.dtau), _rec(blk.q), _dimsOf(blk))


lib.pgDTrk3s2.argtypes = [_view, _view, _view, _view, _dims]


def DTrk3s2(blk):
    lib.pgDTrk3s2(_rec(blk.Q0), _rec(blk.dQ), _rec(blk.dtau), _rec(blk.q), _dimsOf(blk))


lib.pgDTrk3s3.argtypes = [_view, _view, _view, _view, _dims]


def DTrk3s3(blk):
    lib.pgDTrk3s3(_rec(blk.Q0), _rec(blk.dQ), _rec(blk.dtau), _rec(blk.q), _dimsOf(blk))


lib.pgInvertDQ.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    ctypes.c_double,
    _dims,
    ctypes.c_double,
    ctypes.c_bool,
]


def invertDQ(blk, th, dt, viscous):
    lib.pgInvertDQ(
        _rec(blk.Q),
        _rec(blk.dIJK),
        _rec(blk.dQ),
        _rec(blk.dtau),
        _rec(blk.q),
        _rec(blk.qh),
        _rec(blk.qt),
        _rec(th.MW),
        th.Ru,
        _dimsOf(blk),
        dt,
        viscous,
    )


lib.pgResidual.argtypes = [
    _view,
    _view,
    _dims,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def residual(blk):
    """(max, sum of squares) of every primitive's residual over the block."""
    ne = blk.ne
    rMax, rSum = np.zeros(ne), np.zeros(ne)
    lib.pgResidual(
        _rec(blk.q), _rec(blk.Q0), _dimsOf(blk), _doubles(rMax), _doubles(rSum)
    )
    return rMax, rSum
