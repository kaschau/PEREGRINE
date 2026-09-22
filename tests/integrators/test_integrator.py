"""An integrator holds its controller and declares only what it needs;
dual time keeps its two earlier states in two banks and never copies the
newest over the older."""

import numpy as np
import pytest

import peregrinepy as pg

from ..gases import configure


def solver(integrator="rk3", controller="fixed", **settings):
    """A viscous periodic box; each setting goes to the section that has
    it, the step sizing's or the integrator's."""
    config = pg.files.configFile()
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["integrator"] = integrator
    config["simulation"]["controller"] = controller
    config["simulation"]["dt"] = 1e-6
    for key, value in settings.items():
        section = "simulation" if key in config["simulation"] else "timeIntegration"
        config[section][key] = value
    config["initialConditions"]["u"] = 10.0
    configure(config, "air", "navierStokes")
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )
    return pg.multiBlock.solver(config, mesh)


def test_integratorOwnsItsController(my_setup):
    mb = solver("rk3", "cfl", maxCFL=0.5, maxDt=1e-3)
    integrator = mb.integrator
    assert type(integrator.controller).name == "cfl"
    # the controller's kernel is the case's, through the integrator
    assert "CFLmax" in integrator.kernels and "CFLmax" in mb.kernels
    assert "CFLmax" not in solver("rk3", "fixed").kernels
    dt = integrator.controller.stepSize()
    assert 0 < dt <= 1e-3


@pytest.mark.parametrize(
    "integrator,storage", [("rk1", ()), ("rk3", ("Q0",)), ("rk4", ("Q0", "Q1"))]
)
def test_storageIsDeclaredOnTheBlocks(my_setup, integrator, storage):
    mb = solver(integrator)
    assert mb.integrator.storage == storage
    for name in storage:
        assert (
            name in mb.arrays
            and getattr(mb.blocks[0], name).shape == mb.blocks[0].Q.shape
        )
    assert len(mb.graphs["combine"]) == len(mb.integrator.stages)


def test_dualTimeBanksTheEarlierStates(my_setup):
    mb = solver("dualTime", subIterations=3)
    integrator, blk = mb.integrator, mb.blocks[0]
    assert set(mb.graphs) >= {"localDtau", "pseudo 0", "pseudo 1", "combine"}
    assert integrator.restartArrays == ("Qnm1",)
    Qn, Qnm1 = blk.Qn, blk.Qnm1
    assert integrator.bank == 0
    Q0 = blk.Q.get()
    integrator.step(1e-6)
    # the names traded places: nothing was copied over the newest state
    assert blk.Qn is Qnm1 and blk.Qnm1 is Qn
    assert integrator.bank == 1
    # Qn is this state, Qnm1 the one before
    assert np.array_equal(blk.Qn.get(), blk.Q.get())
    assert np.array_equal(blk.Qnm1.get(), Q0)
    integrator.step(1e-6)
    assert blk.Qn is Qn and integrator.bank == 0
    assert np.array_equal(blk.Qn.get(), blk.Q.get())


def test_dualTimeRestoresWhatAResultCarried(my_setup):
    mb = solver("dualTime", subIterations=2)
    integrator, blk = mb.integrator, mb.blocks[0]
    before = blk.Q.get()
    Qnm1 = before * 0.5
    blk.Qnm1.set(Qnm1)
    integrator.restore(found={"Qnm1"})
    assert np.array_equal(blk.Qnm1.get(), Qnm1)
    assert np.array_equal(blk.Qn.get(), before)
    integrator.restore(found=set())
    assert np.array_equal(blk.Qnm1.get(), before)


def test_reportSaysWhatItIs(my_setup):
    mb = solver("dualTime", "fixed")
    text = mb.integrator.report()
    assert "dualTime" in text and "fixed" in text and "rk3" in text
    assert "Low-Mach Preconditioning: on" in text and "Chemistry Jacobian: none" in text
    assert mb.integrator.stepReport() is None
    mb.integrator.reportDue = True
    mb.integrator.step(1e-6)
    assert "SubIter" in mb.integrator.stepReport()


def test_dualTimeBakesItsPseudoSystem(my_setup):
    # the preconditioner is a define on the pseudo kernels; a Jacobian names
    # its rung and forces the chemistry header in
    k = solver("dualTime").kernels
    assert "PG_LOW_MACH=1" in k["invertDQ"].defines
    assert "PG_LOW_MACH=1" in k["localDtau"].defines
    assert k["invertDQ"].includes == ()
    off = solver("dualTime", lowMach=False)
    assert "PG_LOW_MACH=1" not in off.kernels["invertDQ"].defines
    assert "PG_LOW_MACH=1" not in off.kernels["localDtau"].defines
    assert "Low-Mach Preconditioning: off" in off.integrator.report()


def test_dualTimeRefusesAJacobianItCannotUse(my_setup):
    from peregrinepy.files.configFile import pgConfigError

    # no chemistry to differentiate, an integrator without a pseudo system,
    # a rung that is not one, a preconditioner that is neither on nor off
    with pytest.raises(pgConfigError):
        solver("dualTime", chemistryJacobian="diagonal")
    with pytest.raises(pgConfigError):
        solver("rk3", chemistryJacobian="diagonal")
    with pytest.raises(pgConfigError):
        solver("dualTime", chemistryJacobian="full")
    with pytest.raises(pgConfigError):
        solver("dualTime", lowMach="yes")
