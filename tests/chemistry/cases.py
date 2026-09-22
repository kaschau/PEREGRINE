"""One reacting cell: a case of the named mechanism with the chemistry the
config names, its state set, its right-hand side run once and the net
production rates of its state kept beside it; and the ignition of the
stoichiometric CH4/O2 mixture, ours and Cantera's, to compare."""

from pathlib import Path

import cantera as ct
import numpy as np

import peregrinepy as pg
from peregrinepy.kernel import CellCenterKernel
from peregrinepy.multiBlock.arrays import CellCenterArray

ct.add_directory(
    str(Path(__file__).parent / "../../src/peregrinepy/mixture/database/mechanisms")
)
mechanism = "CH4_O2_FFCMY.yaml"


class withRates(pg.multiBlock.solver):
    """A solver keeping the production rates of the state, which the step
    does not: the standalone kernel and its array declared."""

    def _declArrays(self):
        super()._declArrays()
        self.declArray("omega", CellCenterArray, components=self.simulator.mixture.ns)

    def _declKernels(self):
        super()._declKernels()
        self.kernels["productionRates"] = CellCenterKernel(
            "chemistry/productionRates.cpp"
        )


def cell(
    mechanism,
    chemistry,
    p,
    T,
    Y,
    dt=1e-9,
    maxSubSteps=200,
    precision="double",
    solver=withRates,
    cells=1,
    **timeIntegration,
):
    """Makes a reacting case of a uniform cube of :cells: interior cells a
    side at (p, T, Y) with the time integration settings given, runs its
    right-hand side and takes its production rates; gives the solver, of
    the class asked for. One cell has no length to size a pseudo step by;
    dual time takes more."""
    config = pg.files.configFile()
    config["timeIntegration"].update(timeIntegration)
    sim = config["simulation"]
    sim["simulator"] = "euler"
    sim["precision"] = precision
    sim["dt"] = dt
    mixture = config["mixture"]
    mixture["species"] = mechanism
    mixture["eos"] = "tpg"
    mixture["Trange"] = (300.0, 3500.0)
    config["chemistry"]["source"] = chemistry
    config["chemistry"]["maxSubSteps"] = maxSubSteps
    config["RHS"]["primaryAdvFlux"] = "KEPaEC"
    config["bcValues"]["walls"] = {"bcType": "adiabaticSlipWall"}
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1],
        dimsPerBlock=[cells + 1] * 3,
        lengths=[0.01, 0.01, 0.01],
        boundaryNames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = solver(config, mesh)
    blk = mb.blocks[0]
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[..., 0], q[..., 4], q[..., 5:] = p, T, Y[:-1]
    mb.setPrimitives([q])
    mb.integrator.initialize()
    mb.integrator.dtOnDevice.set([dt])
    mb.rhs()
    mb.launch("productionRates", "interior")
    return mb


def at(mb, name):
    """The named array at the one interior cell."""
    ng = mb.ng
    return getattr(mb.blocks[0], name).get()[ng, ng, ng]


def stoichiometric():
    """The stoichiometric CH4/O2 mixture's mass fractions."""
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1.0, "CH4", "O2")
    return gas.Y


def canterasDelay(p0, T0, Y, rise):
    """The time Cantera's constant-volume reactor takes to T0 + rise."""
    gas = ct.Solution(mechanism)
    gas.TPY = T0, p0, Y
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])
    t = 0.0
    while reactor.T < T0 + rise:
        t = net.step()
    return t


def oursDelay(mb, dt, T0, rise, until):
    """The time the case takes to T0 + rise stepping by dt, None if it has
    not by :until:."""
    while mb.tme < until:
        mb.integrator.step(dt)
        if at(mb, "q")[1] > T0 + rise:
            return mb.tme
    return None


def randomState(mixture, rng):
    """A random composition over the species, and a temperature and
    pressure inside the case's range."""
    Y = rng.random(mixture.ns)
    return rng.uniform(1e5, 30e5), rng.uniform(600.0, 2800.0), Y / Y.sum()
