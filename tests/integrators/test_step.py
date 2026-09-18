"""Every integrator, with either controller, steps a uniform flow through
a periodic box and leaves it as it was, for either physics; dual time does
so with any Runge-Kutta scheme as its pseudo time scheme."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.files.configFile import pgConfigError

from ..gases import configure


def uniformBox(integrator, controller, physics):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["controller"] = controller
    config["timeIntegration"]["dt"] = 1e-6
    config["initialConditions"]["u"] = 10.0
    configure(config, "air", physics)
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )
    return config, mesh


def assertUnchanged(mb, name, before):
    blk = mb.blocks[0]
    after = getattr(blk, name).get()
    ng = blk.ng
    assert np.isfinite(after).all()
    # against each component's own size; the transverse momenta are zero
    scale = np.abs(before[ng, ng, ng]).max()
    interior = np.s_[ng:-ng, ng:-ng, ng:-ng]
    assert np.abs(after[interior] - before[interior]).max() < 1e-10 * scale


@pytest.mark.parametrize(
    "integrator", ["rk1", "rk2", "rk3", "rk34", "rk4", "maccormack", "dualTime"]
)
@pytest.mark.parametrize("physics", ["euler", "navierStokes"])
@pytest.mark.parametrize("controller", ["fixed", "cfl"])
def test_step(my_setup, integrator, physics, controller):
    config, mesh = uniformBox(integrator, controller, physics)

    # dual time converges each step in pseudo time; a step sized by the CFL is refused
    if integrator == "dualTime" and controller == "cfl":
        with pytest.raises(pgConfigError):
            pg.multiBlock.solver(config, mesh)
        return
    mb = pg.multiBlock.solver(config, mesh)
    assert type(mb.integrator).__name__ == integrator
    assert type(mb.integrator.controller).controllerName == controller
    assert type(mb.simulation).physics == physics
    assert mb.integrator.controller.stepSize() > 0.0

    Q0 = mb.blocks[0].Q.get()
    for _ in range(3):
        mb.integrator.step(config["timeIntegration"]["dt"])
    assertUnchanged(mb, "Q", Q0)
    assert mb.nrt == 3


@pytest.mark.parametrize("pseudo", ["rk1", "rk2", "rk3", "rk34", "rk4", "maccormack"])
def test_dualTimeComposesItsPseudoScheme(my_setup, pseudo):
    config, mesh = uniformBox("dualTime", "fixed", "navierStokes")
    config["timeIntegration"]["pseudoIntegrator"] = pseudo
    config["timeIntegration"]["subIterations"] = 4
    mb = pg.multiBlock.solver(config, mesh)
    integrator = mb.integrator
    assert integrator.pseudo.integratorName == pseudo
    assert integrator.subIterations == 4
    # the pseudo scheme's storage joins dual time's own
    assert ("Q0" in mb.arrays) == (pseudo != "rk1")
    assert ("Q1" in mb.arrays) == (pseudo == "rk4")
    assert {"Qn", "Qnm1", "dtau"} <= set(mb.arrays)
    assert pseudo in repr(mb)

    q0 = mb.blocks[0].q.get()
    for _ in range(2):
        integrator.step(config["timeIntegration"]["dt"])
    assertUnchanged(mb, "q", q0)

    config["timeIntegration"]["pseudoIntegrator"] = "dualTime"
    with pytest.raises(pgConfigError):
        pg.multiBlock.solver(config, mesh)
