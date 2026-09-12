"""The boundary conditions: every one takes the same views, and python runs
the eos on the face's halo after the euler terms, which is what the C++ used
to call back for."""

import ctypes

from ..abi import Dims, View, lib

_view, _dims = ctypes.POINTER(View), ctypes.POINTER(Dims)
_terms = {"euler": 0, "preDqDxyz": 1, "postDqDxyz": 2}
# the area vectors of the faces a boundary face's normal crosses
_areaVectors = {1: "iS", 2: "iS", 3: "jS", 4: "jS", 5: "kS", 6: "kS"}


def _rec(array):
    return ctypes.byref(View.of(array))


def _call(name, blk, face, stage, tme):
    getattr(lib, name)(
        _rec(blk.q),
        _rec(blk.Q),
        _rec(blk.qh),
        # a case without diffusion has no gradients, and its stages never read them
        _rec(getattr(blk, "grads", None)),
        _rec(getattr(blk, _areaVectors[face.nface])),
        _rec(face.qBcVals),
        _rec(face.QBcVals),
        _rec(face.periodicRotMatrix),
        ctypes.byref(Dims.of(blk)),
        face.nface,
        stage,
        tme,
    )


def _boundaryCondition(name):
    cname = "pg" + name[0].upper() + name[1:]
    lib.declare(
        cname, [_view] * 8 + [_dims, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    )

    def kernel(blk, face, eos, th, terms, tme):
        _call(cname, blk, face, _terms[terms], tme)
        if terms == "euler":
            eos(blk, th, face.nface, "prims")
            # the mass flux inlet sets its velocities once the halo has a density
            if name == "constantMassFluxSubsonicInlet":
                _call(cname, blk, face, 3, tme)
                eos(blk, th, face.nface, "prims")

    kernel.__name__ = name
    return kernel


walls = {
    n: _boundaryCondition(n)
    for n in (
        "adiabaticNoSlipWall",
        "adiabaticSlipWall",
        "adiabaticMovingWall",
        "isoTNoSlipWall",
        "isoTSlipWall",
        "isoTMovingWall",
    )
}
inlets = {
    n: _boundaryCondition(n)
    for n in (
        "constantVelocitySubsonicInlet",
        "supersonicInlet",
        "constantMassFluxSubsonicInlet",
        "stagnationSubsonicInlet",
    )
}
exits = {
    n: _boundaryCondition(n) for n in ("constantPressureSubsonicExit", "supersonicExit")
}
periodics = {n: _boundaryCondition(n) for n in ("periodicRot",)}
