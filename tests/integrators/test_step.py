"""Every stepper, composed with either controller, steps a uniform flow
through a periodic box and leaves it as it was; dual time does so with any
Runge-Kutta stepper as its pseudo time scheme."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.files.configFile import pgConfigError

from ..gases import configure


@pytest.mark.parametrize(
    "integrator", ["rk1", "rk2", "rk3", "rk34", "rk4", "maccormack", "dualTime"]
)
@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize("controller", ["fixed", "cfl"])
def test_step(my_setup, integrator, diffusion, controller):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["RHS"]["diffusion"] = diffusion
    config["timeIntegration"]["integrator"] = integrator
    config["timeIntegration"]["controller"] = controller
    config["timeIntegration"]["dt"] = 1e-6
    config["initialConditions"]["u"] = 10.0
    configure(config, "air")
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )

    # dual time converges each step in pseudo time; a step sized by the CFL is refused
    if integrator == "dualTime" and controller == "cfl":
        with pytest.raises(pgConfigError):
            pg.integrators.getSolver(config, mesh=mesh)
        return
    mb = pg.integrators.getSolver(config, mesh=mesh)
    assert (
        type(mb).__name__
        == f"{integrator}_{controller.upper() if controller == 'cfl' else 'Fixed'}_solver"
    )
    assert mb.stepSize() > 0.0

    blk = mb.blocks[0]
    Q0 = blk.Q.get()
    for _ in range(3):
        mb.step(config["timeIntegration"]["dt"])
    Q = blk.Q.get()
    ng = blk.ng
    assert np.isfinite(Q).all()
    # against each component's own size; the transverse momenta are zero
    scale = np.abs(Q0[ng, ng, ng]).max()
    assert (
        np.abs(Q[ng:-ng, ng:-ng, ng:-ng] - Q0[ng:-ng, ng:-ng, ng:-ng]).max()
        < 1e-10 * scale
    )
    assert mb.nrt == 3


@pytest.mark.parametrize("pseudo", ["rk1", "rk2", "rk3", "rk34", "rk4", "maccormack"])
def test_dualTimeComposesItsPseudoScheme(my_setup, pseudo):
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = "dualTime"
    config["timeIntegration"]["pseudoIntegrator"] = pseudo
    config["timeIntegration"]["subIterations"] = 4
    config["timeIntegration"]["dt"] = 1e-6
    config["initialConditions"]["u"] = 10.0
    configure(config, "air")
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )
    mb = pg.integrators.getSolver(config, mesh=mesh)
    assert mb.pseudo.stepperName == pseudo
    assert mb.subIterations == 4
    # the pseudo scheme's storage joins dual time's own
    assert ("Q0" in mb.arrays) == (pseudo != "rk1")
    assert ("Q1" in mb.arrays) == (pseudo == "rk4")
    assert {"Qn", "Qnm1", "dtau"} <= set(mb.arrays)
    assert pseudo in repr(mb)

    blk = mb.blocks[0]
    q0 = blk.q.get()
    for _ in range(2):
        mb.step(config["timeIntegration"]["dt"])
    q = blk.q.get()
    ng = blk.ng
    assert np.isfinite(q).all()
    scale = np.abs(q0[ng, ng, ng]).max()
    assert (
        np.abs(q[ng:-ng, ng:-ng, ng:-ng] - q0[ng:-ng, ng:-ng, ng:-ng]).max()
        < 1e-10 * scale
    )

    config["timeIntegration"]["pseudoIntegrator"] = "dualTime"
    with pytest.raises(pgConfigError):
        pg.integrators.getSolver(config, mesh=mesh)
