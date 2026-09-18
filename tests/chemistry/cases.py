"""One reacting cell: a case of the named mechanism with the chemistry the
config names, its state set, and its right-hand side run once."""

import numpy as np

import peregrinepy as pg


def cell(mechanism, chemistry, p, T, Y, dt=1e-9, maxSubSteps=10, precision="double"):
    """Makes a reacting case of one interior cell at (p, T, Y) and runs its
    right-hand side; gives the solver."""
    config = pg.files.configFile()
    sim = config["simulation"]
    sim["physics"] = "euler"
    sim["mixture"] = mechanism
    sim["eos"] = "tpg"
    sim["Trange"] = (300.0, 3500.0)
    sim["precision"] = precision
    sim["chemistry"] = chemistry
    sim["chemistryMaxSubSteps"] = maxSubSteps
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["timeIntegration"]["dt"] = dt
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[2, 2, 2],
        lengths=[1, 1, 1],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.multiBlock.solver(config, mesh)
    blk = mb.blocks[0]
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[..., 0], q[..., 4], q[..., 5:] = p, T, Y[:-1]
    mb.setPrimitives([q])
    mb.integrator.initialize()
    mb.integrator.dtOnDevice.set([dt])
    mb.rhs()
    return mb


def at(mb, name):
    """The named array at the one interior cell."""
    ng = mb.ng
    return getattr(mb.blocks[0], name).get()[ng, ng, ng]


def randomState(mixture, rng):
    """A random composition over the species, and a temperature and
    pressure inside the case's range."""
    Y = rng.random(mixture.ns)
    return rng.uniform(1e5, 30e5), rng.uniform(600.0, 2800.0), Y / Y.sum()
