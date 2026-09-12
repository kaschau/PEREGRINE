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


lib.pgDQzero.argtypes = [_view, _dims]
lib.pgDq2FD.argtypes = [_view, _view, _view, _dims]
lib.pgViscousSponge.argtypes = [
    _view,
    _view,
    _dims,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
]


def dQzero(blk):
    lib.pgDQzero(_rec(blk.dQ), _dimsOf(blk))


def dq2FD(blk):
    lib.pgDq2FD(_rec(blk.dENCdxyz), _rec(blk.grads), _rec(blk.q), _dimsOf(blk))


def viscousSponge(blk, origin, ending, mult):
    lib.pgViscousSponge(
        _rec(blk.cells),
        _rec(blk.qt),
        _dimsOf(blk),
        _doubles(origin),
        _doubles(ending),
        mult,
    )


lib.pgApplyFlux.argtypes = [_view, _view, _view, _view, _view, _dims]


def applyFlux(blk, weight=None):
    lib.pgApplyFlux(
        _rec(blk.J),
        _rec(blk.dQ),
        _rec(blk.iF),
        _rec(blk.jF),
        _rec(blk.kF),
        _dimsOf(blk),
    )


lib.pgApplyHybridFlux.argtypes = [
    _view,
    _view,
    _view,
    _view,
    _view,
    _view,
    _dims,
    ctypes.c_double,
]


def applyHybridFlux(blk, primary):
    lib.pgApplyHybridFlux(
        _rec(blk.J),
        _rec(blk.dQ),
        _rec(blk.iF),
        _rec(blk.jF),
        _rec(blk.kF),
        _rec(blk.phi),
        _dimsOf(blk),
        primary,
    )


lib.pgApplyDissipationFlux.argtypes = [_view, _view, _view, _view, _view, _dims]


def applyDissipationFlux(blk, weight=None):
    lib.pgApplyDissipationFlux(
        _rec(blk.J),
        _rec(blk.dQ),
        _rec(blk.iF),
        _rec(blk.jF),
        _rec(blk.kF),
        _dimsOf(blk),
    )


lib.pgAllFinite.argtypes = [_view, _dims]
lib.pgAllFinite.restype = ctypes.c_int
lib.pgCFLmax.argtypes = [_view] * 6 + [_dims, ctypes.POINTER(ctypes.c_double)]


def allFinite(blk):
    """Whether every conserved quantity in the interior is finite."""
    return bool(lib.pgAllFinite(_rec(blk.Q), _dimsOf(blk)))


def CFLmax(blk):
    """The block's max acoustic, convective and combined CFL speeds."""
    cfl = np.zeros(3)
    lib.pgCFLmax(
        _rec(blk.dIJK),
        _rec(blk.iS),
        _rec(blk.jS),
        _rec(blk.kS),
        _rec(blk.q),
        _rec(blk.qh),
        _dimsOf(blk),
        _doubles(cfl),
    )
    return cfl


lib.pgAxpby.argtypes = [_view, ctypes.c_double, ctypes.c_double, _view]
lib.pgAxpbypcz.argtypes = [
    _view,
    ctypes.c_double,
    ctypes.c_double,
    _view,
    ctypes.c_double,
    _view,
]
lib.pgExtractSendBuffer.argtypes = [
    _view,
    _view,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
] + [ctypes.c_int] * 3
lib.pgPlaceRecvBuffer.argtypes = [
    _view,
    _view,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]


def AEQB(A, B):
    """A = B, for two arrays of the same shape."""
    lib.pgAxpby(_rec(A), 0.0, 1.0, _rec(B))


def axnpby(A, a, b, B, c=None, C=None):
    """A = a*A + b*B [+ c*C]."""
    if C is None:
        lib.pgAxpby(_rec(A), a, b, _rec(B))
    else:
        lib.pgAxpbypcz(_rec(A), a, b, _rec(B), c, _rec(C))


def _ints(values):
    return np.ascontiguousarray(values, dtype=np.int32).ctypes.data_as(
        ctypes.POINTER(ctypes.c_int)
    )


def extractSendBuffer(view, buffer, face, slices):
    """Pack the face's send buffer from the view, turned onto the neighbor's frame."""
    lib.pgExtractSendBuffer(
        _rec(view),
        _rec(buffer),
        face.nface,
        _ints(slices),
        len(slices),
        int(face._transposed),
        int(0 in face._flipped),
        int(1 in face._flipped),
    )


def placeRecvBuffer(view, buffer, face, slices):
    """Unpack the face's receive buffer into the view."""
    lib.pgPlaceRecvBuffer(
        _rec(view), _rec(buffer), face.nface, _ints(slices), len(slices)
    )
