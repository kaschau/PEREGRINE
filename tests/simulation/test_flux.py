"""An advective flux is composed from the scheme the config names -- a
Riemann solver alone, or reconstruct-limiter-riemann -- and each
composition is the order it claims on an advected density wave."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.kernel import FluxKernel

air = {"Air": {"MW": 28.97, "cp0": 1000.0}}
R = 8314.46261815324 / air["Air"]["MW"]


def wave(scheme, nx):
    """The max error of a density wave advected once around a periodic
    line of :nx: - 1 cells."""
    config = pg.files.configFile()
    config["simulation"]["physics"] = "euler"
    config["simulation"]["mixture"] = air
    config["RHS"]["primaryAdvFlux"] = scheme
    config["timeIntegration"]["integrator"] = "rk3"
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[1, 0.1, 0.1],
        periodic=[True, False, False],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.multiBlock.solver(config, mesh)
    blk = mb.blocks[0]
    ng = blk.ng
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[..., 0] = 1.0
    q[..., 1] = 1.0
    xc = blk.cells.get()[..., 0]
    q[..., 4] = 1.0 / (R * (2.0 + np.sin(2 * np.pi * xc)))
    mb.setPrimitives([q])
    dt = 0.1 / (nx - 1)
    n = int(round(1.0 / dt))
    for _ in range(n):
        mb.integrator.step(dt)
    exact = 2.0 + np.sin(2 * np.pi * (xc[ng:-ng, ng, ng] - n * dt))
    return np.abs(blk.Q.get()[ng:-ng, ng, ng, 0] - exact).max()


def test_theSchemeStringComposes():
    k = FluxKernel("muscl-vanLeer-rusanov", 0)
    assert k.stencil == 2 and k.__name__ == "muscl-vanLeer-rusanov"
    assert "PG_RIEMANN=rusanov" in k.defines and "PG_LIMITER=vanLeer" in k.defines
    assert k.includes[0] == "advFlux/limiter/vanLeer.hpp"
    first = FluxKernel("hllc", 1)
    assert first.stencil == 1 and "PG_RECONSTRUCT=piecewiseConstant" in first.defines
    assert FluxKernel.composed("ausmPlusUp") and not FluxKernel.composed("KEPaEC")
    with pytest.raises(ValueError):
        FluxKernel("muscl-rusanov", 0)


@pytest.mark.parametrize("riemann", ["rusanov", "hllc", "ausmPlusUp"])
def test_musclIsSharperThanPiecewiseConstant(my_setup, riemann):
    first = wave(riemann, 41)
    second = wave(f"muscl-vanLeer-{riemann}", 41)
    assert second < 0.25 * first


@pytest.mark.parametrize(
    "limiter,atLeast",
    [("minmod", 1.0), ("vanLeer", 1.3), ("mc", 1.5), ("superbee", 1.3)],
)
def test_limitersOrderOnTheWave(my_setup, limiter, atLeast):
    # the max norm, where a limiter clips the crests: under two, above one
    e40, e80 = wave(f"muscl-{limiter}-rusanov", 41), wave(
        f"muscl-{limiter}-rusanov", 81
    )
    assert np.log2(e40 / e80) > atLeast
