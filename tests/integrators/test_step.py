"""Every integrator, with either controller, steps a uniform flow through
a periodic box and leaves it as it was, for either physics; dual time does
so with any Runge-Kutta scheme as its pseudo time scheme."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.files.configFile import pgConfigError

from ..gases import configure, primitives


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
    assert type(mb.integrator.controller).name == controller
    assert type(mb.simulator).name == physics
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
    assert integrator.pseudo.name == pseudo
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


@pytest.mark.parametrize("eos", ["tpg", "realGas"])
def test_dualTimeOnAnyEos(my_setup, eos):
    # a perturbed dense cold state, far from ideal for the cubic: the pseudo
    # iterations converge the step and the state stays finite and near
    config, mesh = uniformBox("dualTime", "fixed", "euler")
    config["simulation"]["mixture"] = ["O2", "N2", "CO2", "CH4"]
    config["simulation"]["eos"] = eos
    config["simulation"]["trans"] = None
    config["simulation"]["Trange"] = (300.0, 3500.0)
    config["initialConditions"].update(
        p=60e5, T=320.0, Y={"O2": 0.3, "N2": 0.4, "CO2": 0.2}
    )
    config["timeIntegration"]["subIterations"] = 8
    config["timeIntegration"]["dt"] = 1e-7
    mb = pg.multiBlock.solver(config, mesh)
    blk = mb.blocks[0]
    ng = blk.ng
    rng = np.random.default_rng(3)
    # a slight pressure perturbation, so the pseudo iterations have work
    q = primitives(mb, blk)
    q[..., 0] *= 1 + 0.01 * rng.random(q.shape[:3])
    mb.setPrimitives([q])
    mb.integrator.initialize()
    mb.integrator.reportDue = True
    for _ in range(2):
        mb.integrator.step(config["timeIntegration"]["dt"])
    residuals = np.array(mb.integrator.residuals)
    assert np.isfinite(blk.Q.get()[ng:-ng, ng:-ng, ng:-ng]).all()
    # the pseudo residual fell over the sub iterations of the last step
    assert residuals[-1, 0] < 0.5 * residuals[0, 0]
