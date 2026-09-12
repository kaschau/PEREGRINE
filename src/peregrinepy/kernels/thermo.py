"""The equations of state: conserved from primitive, or back."""

import ctypes

from ..abi import Range, View, lib

_view = ctypes.POINTER(View)
_eosArgs = [_view] * 4
lib.declare(
    "pgCpg",
    _eosArgs
    + [
        _view,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(Range),
    ],
)
lib.declare(
    "pgTpg",
    (_eosArgs + [_view] * 3 + [ctypes.c_double, ctypes.c_int, ctypes.POINTER(Range)]),
)
lib.declare(
    "pgRealGas",
    (_eosArgs + [_view] * 6 + [ctypes.c_double, ctypes.c_int, ctypes.POINTER(Range)]),
)


def _views(blk, th, names):
    arrays = [blk.Q, blk.q, blk.qh] + [getattr(th, n) for n in ["MW"] + names]
    return [ctypes.byref(View.of(a)) for a in arrays]


def cpg(blk, th, nface, given):
    lib.pgCpg(
        *_views(blk, th, ["cp0"]),
        th.Ru,
        given == "prims",
        ctypes.byref(Range.of(blk, nface)),
    )


def tpg(blk, th, nface, given):
    lib.pgTpg(
        *_views(blk, th, ["cpPoly", "hPoly", "hRef"]),
        th.Ru,
        given == "prims",
        ctypes.byref(Range.of(blk, nface)),
    )


def realGas(blk, th, nface, given):
    lib.pgRealGas(
        *_views(blk, th, ["cpPoly", "hPoly", "hRef", "Tcrit", "pcrit", "acentric"]),
        th.Ru,
        given == "prims",
        ctypes.byref(Range.of(blk, nface)),
    )
