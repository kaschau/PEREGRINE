"""An advective flux is composed from the scheme the config names -- a
Riemann solver alone, reconstruct-limiter-riemann, or a central scheme,
with a secondary blended in by a switch for shock capturing -- and each
composition is the order it claims on an advected density wave."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.kernel import FluxKernel

from ..riemannProblem import RiemannProblem

air = {"Air": {"MW": 28.97, "cp0": 1000.0, "mu0": 1.86e-5, "kappa0": 0.0263}}
R = 8314.46261815324 / air["Air"]["MW"]
gamma = air["Air"]["cp0"] / (air["Air"]["cp0"] - R)


def line(
    scheme,
    nx,
    secondary=None,
    switch=None,
    values=(),
    physics="euler",
    periodic=True,
    configure=None,
):
    """A case on a line of :nx: - 1 cells with the named fluxes; :configure:
    edits the config before the case is made."""
    config = pg.files.configFile()
    config["simulation"]["physics"] = physics
    config["simulation"]["mixture"] = air
    config["simulation"]["trans"] = (
        "constantProps" if physics == "navierStokes" else None
    )
    config["RHS"]["primaryAdvFlux"] = scheme
    config["RHS"]["secondaryAdvFlux"] = secondary
    config["RHS"]["switchAdvFlux"] = switch
    config["RHS"]["switchValues"] = dict(values)
    config["timeIntegration"]["integrator"] = "rk3"
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    if configure:
        configure(config)
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[nx, 2, 2],
        lengths=[1, 0.1, 0.1],
        periodic=[periodic, False, False],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    return pg.multiBlock.solver(config, mesh)


def wave(scheme, nx, **fluxes):
    """The max error of a density wave advected once around a periodic
    line of :nx: - 1 cells."""
    mb = line(scheme, nx, **fluxes)
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


def sod(scheme, **fluxes):
    """The mean density error along Sod's tube at t = 0.2 on 200 cells,
    against the exact solution."""
    test = RiemannProblem.toro(0, gamma=gamma, R=R)
    mb = line(scheme, 201, periodic=False, **fluxes)
    blk = mb.blocks[0]
    ng = blk.ng
    xc = blk.cells.get()[..., 0]
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[..., 0] = np.where(xc <= test.x0, test.pL, test.pR)
    q[..., 4] = np.where(xc <= test.x0, test.TL, test.TR)
    mb.setPrimitives([q])
    for _ in range(int(round(test.t / test.dt))):
        mb.integrator.step(test.dt)
    x = xc[ng:-ng, ng, ng]
    rho = mb.exportData(blk, ["rho"])["rho"][ng:-ng, ng, ng]
    return np.abs(rho - test.solve(x)["rho"]).mean()


def test_theSchemeStringComposes():
    k = FluxKernel("muscl-vanLeer-rusanov", 0)
    assert k.stencil == 2 and k.__name__ == "muscl-vanLeer-rusanov"
    assert "PG_PRIMARY=rusanov" in k.defines and "PG_LIMITER=vanLeer" in k.defines
    assert "PG_BASE=muscl" in k.defines
    assert k.includes[0] == "advFlux/limiter/vanLeer.hpp"
    first = FluxKernel("hllc", 1)
    assert first.stencil == 1 and "PG_RECONSTRUCT=piecewiseConstant" in first.defines
    assert FluxKernel.composed("ausmPlusUp") and FluxKernel.composed("KEPaEC")
    fourth = FluxKernel("fourthOrderKEPaEC", 2)
    assert fourth.stencil == 2 and "PG_RECONSTRUCT=fourCells" in fourth.defines
    with pytest.raises(ValueError):
        FluxKernel("muscl-rusanov", 0)


def test_theSwitchIsTheBaseAndWidensTheStencil():
    k = FluxKernel("KEPaEC", 0, "rusanov", "jamesonPressure", {"gain": 5})
    assert k.stencil == 2 and "PG_BASE=jamesonPressure" in k.defines
    assert "PG_SECONDARY=rusanov" in k.defines
    assert k.includes[-1] == "advFlux/switch/jamesonPressure.hpp"
    assert k.__name__ == "KEPaEC with rusanov by jamesonPressure"
    # the ducros switch's columns come in through it, its values baked
    d = FluxKernel("KEPaEC", 0, "rusanov", "ducros", {"nu": 0.1, "floor": 0.01})
    assert d.stencil == 1 and "PG_DUCROS_NU=0.1" in d.defines
    names = [n for _, n, _, _ in d.structs["k"][1]]
    assert "grads" in names and "cellLength" in names
    with pytest.raises(ValueError):
        FluxKernel("KEPaEC", 0, "rusanov")


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


def test_aNearZeroWeightLeavesThePrimary(my_setup):
    # the wave's pressure and velocity are uniform to rounding, so neither
    # switch fires beyond it, and the blend is the primary's flux
    alone = wave("KEPaEC", 41)
    blended = wave(
        "KEPaEC",
        41,
        secondary="rusanov",
        switch="jamesonPressure",
        values={"gain": 5.0},
    )
    assert abs(blended - alone) < 1e-12 * alone
    viscous = wave("KEPaEC", 41, physics="navierStokes")
    ducros = wave(
        "KEPaEC",
        41,
        secondary="rusanov",
        switch="ducros",
        values={"nu": 0.1, "floor": 0.0},
        physics="navierStokes",
    )
    # conduction across the wave moves the velocity a little, and ducros sees it
    assert abs(ducros - viscous) < 1e-6 * viscous


def test_fourthOrderKEPaECIsFourthOrder(my_setup):
    # measured 1.27e-4, 7.96e-6, 4.98e-7 at 40, 80, 160 cells
    e40, e80 = wave("fourthOrderKEPaEC", 41), wave("fourthOrderKEPaEC", 81)
    assert np.log2(e40 / e80) > 3.8


@pytest.mark.parametrize(
    "primary,switch,values,physics,atMost",
    [
        ("KEPaEC", "jamesonPressure", {"gain": 5.0}, "euler", 0.008),
        ("fourthOrderKEPaEC", "jamesonPressure", {"gain": 5.0}, "euler", 0.006),
        ("KEPaEC", "ducros", {"nu": 0.3, "floor": 0.0}, "navierStokes", 0.011),
    ],
)
def test_theSwitchCapturesTheShock(my_setup, primary, switch, values, physics, atMost):
    # the central scheme alone rings across the tube, about as far from the
    # exact solution as rusanov's smearing; blended by the switch it is
    # sharper than either (measured: KEPaEC 0.0189, rusanov 0.0183, jameson
    # at gain 5 0.0063, fourth order 0.0047, ducros at nu 0.3 0.0089;
    # muscl-vanLeer-rusanov 0.0040)
    alone = sod(primary, physics=physics)
    blended = sod(
        primary, secondary="rusanov", switch=switch, values=values, physics=physics
    )
    assert np.isfinite(blended) and blended < atMost
    assert blended < 0.6 * alone and blended < 0.6 * sod("rusanov", physics=physics)
