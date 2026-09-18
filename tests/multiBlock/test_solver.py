"""The solver is built by interrogating the simulation and the integrator:
the boundaries the mesh names take the config's values, an unnamed side
stays what the mesher made it, a name the config does not know is refused,
the metrics are only what the physics asked for, and the exported data
derives from the state."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.files.configFile import pgConfigError

from ..gases import configure, primitives


def config(physics="navierStokes", gas="air"):
    c = pg.files.configFile()
    configure(c, gas, physics)
    c["RHS"]["primaryAdvFlux"] = "KEPaEC"
    return c


def sided(names={}):
    """A unit cube of 4^3 cells, its sides named by number."""
    return pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[5, 5, 5], lengths=[1, 1, 1], boundaryNames=names
    )


def test_namedBoundariesTakeTheConfigsValues(my_setup):
    c = config()
    c["bcValues"]["inlet"] = {
        "bcType": "constantVelocitySubsonicInlet",
        "u": 3.0,
        "v": 0.0,
        "w": 0.0,
        "T": 310.0,
    }
    c["bcValues"]["exit"] = {"bcType": "constantPressureSubsonicExit", "p": 90000.0}
    mb = pg.multiBlock.solver(c, sided({1: "inlet", 2: "exit"}))
    blk = mb.blocks[0]
    inlet, exit_ = blk.getFace(1), blk.getFace(2)
    assert (
        inlet.bcType == "constantVelocitySubsonicInlet"
        and exit_.bcType == "constantPressureSubsonicExit"
    )
    assert np.all(inlet.qBcVals.get()[..., 4] == 310.0) and np.all(
        inlet.qBcVals.get()[..., 1] == 3.0
    )
    assert np.all(exit_.qBcVals.get()[..., 0] == 90000.0)
    # the sides the mesher did not name are its walls, with no values
    for n in range(3, 7):
        face = blk.getFace(n)
        assert (
            face.bcName is None
            and face.bcType == "adiabaticNoSlipWall"
            and face.qBcVals is None
        )


def test_aNameTheConfigDoesNotKnowIsRefused(my_setup):
    with pytest.raises(pgConfigError):
        pg.multiBlock.solver(config(), sided({1: "mystery"}))
    c = config()
    c["bcValues"]["noType"] = {"p": 1.0}
    with pytest.raises(pgConfigError):
        pg.multiBlock.solver(c, sided({1: "noType"}))


def test_aBoundaryThePhysicsCannotHaveIsRefused(my_setup):
    c = config("euler")
    c["bcValues"]["sticky"] = {"bcType": "adiabaticNoSlipWall"}
    with pytest.raises(KeyError):
        pg.multiBlock.solver(c, sided({1: "sticky"}))


def test_onlyTheMetricsThePhysicsAsksForAreMade(my_setup):
    mb = pg.multiBlock.solver(config("euler"), sided())
    assert set(mb.metrics) == set(mb.simulation.metrics)
    blk = mb.blocks[0]
    # a grid used as a grid asks for none
    grid = pg.multiBlock.grid()
    sided().fill(grid)
    assert grid.metrics == [] and set(grid.arrays) == {"nodes", "cells"}
    # and the volumes of a unit cube of 4^3 cells
    assert np.allclose(1.0 / blk.Jinv.get()[blk.interior], (1 / 4) ** 3)
    with pytest.raises(AttributeError):
        grid.declMetric("volumes")


def test_exportDataDerivesFromTheState(my_setup):
    mb = pg.multiBlock.solver(config(gas="CH4_O2"), sided())
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    rng = np.random.default_rng(1)
    q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
    mb.setPrimitives([q])
    names = mb.exportVars
    data = mb.exportData(blk, names)
    assert list(data) == names and names[0] == "rho"
    Q = blk.Q.get()
    i = blk.interior
    assert np.array_equal(data["rho"][i], Q[i][..., 0])
    assert np.allclose(data["u"][i], Q[i][..., 1] / Q[i][..., 0])
    species = mb.simulation.mixture.speciesNames
    assert np.allclose(sum(data[s][i] for s in species), 1.0)
    # the primitive vector comes back as it was set, on the interior
    assert np.allclose(primitives(mb, blk)[i], q[i])


def test_theCaseReportsItself(my_setup):
    mb = pg.multiBlock.solver(config(), sided())
    text = repr(mb)
    for line in (
        "Blocks: 1 of 1",
        "Physics: navierStokes",
        "Time Integrator: rk3",
        "consistify: pack Q",
    ):
        assert line in text
