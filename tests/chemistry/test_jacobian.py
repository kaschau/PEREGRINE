"""The species source's Jacobian against differences of the rates: the
diagonal rung's entries, each carried species' own d omega_j / d Y_j at
fixed density and temperature with the last species taking up the change,
and its d omega_j / dT at fixed density and composition, at a random state
and at a fresh stoichiometric one, where the radicals are exactly zero and
the log-space form has to give the floor's limit; finite in single."""

import numpy as np

from peregrinepy.kernel import CellCenterKernel
from peregrinepy.multiBlock.arrays import CellCenterArray

from .cases import at, cell, mechanism, randomState, stoichiometric, withRates


class withJacobian(withRates):
    """A solver keeping the source's Jacobian too: the kernel at the
    diagonal rung, and the matrix it fills, a carried species' row over
    the primitives' columns."""

    def _declArrays(self):
        super()._declArrays()
        ns = self.simulator.mixture.ns
        self.declArray("omegaJ", CellCenterArray, components=(ns - 1, self.ne))

    def _declKernels(self):
        super()._declKernels()
        self.kernels["productionRateJacobian"] = CellCenterKernel(
            "chemistry/productionRateJacobian.cpp",
            defines=["PG_CHEMISTRY_JACOBIAN=diagonal"],
        )


def jacobian(mb):
    """The matrix as the kernel fills it at the one cell."""
    mb.launch("productionRateJacobian", "interior")
    return at(mb, "omegaJ")


def rates(mb):
    """The carried species' rates at the cell, of the state as set."""
    mb.launch("productionRates", "interior")
    return at(mb, "omega")[:-1]


def bySpeciesDifferences(mb, step, sides):
    """The species diagonal by differences of the rates between the two
    sides, each carried species' conserved value moved alone by its step:
    the density and the temperature stay, the last species takes up the
    change."""
    blk, ng, ns = mb.blocks[0], mb.ng, mb.simulator.mixture.ns
    Q = blk.Q.get()
    rho = Q[ng, ng, ng, 0]
    d = np.zeros(ns - 1)
    for j in range(ns - 1):
        moved = []
        for side in sides:
            Qs = Q.copy()
            Qs[ng, ng, ng, 5 + j] += side * step[j]
            blk.Q.set(Qs)
            moved.append(rates(mb)[j])
        d[j] = rho * (moved[0] - moved[1]) / ((sides[0] - sides[1]) * step[j])
    blk.Q.set(Q)
    return d


def byTemperatureDifferences(mb, step):
    """The temperature column by central differences of the rates, the
    temperature the rates read moved alone: density and composition
    stay."""
    blk, ng = mb.blocks[0], mb.ng
    q = blk.q.get()
    moved = []
    for side in (1.0, -1.0):
        qs = q.copy()
        qs[ng, ng, ng, 1] += side * step
        blk.q.set(qs)
        moved.append(rates(mb))
    blk.q.set(q)
    return (moved[0] - moved[1]) / (2 * step)


def assertOnlyTheRung(J):
    """The rung leaves every other entry alone."""
    species = J[:, 5:]
    assert not np.any(J[:, :4])
    assert not np.any(species - np.diag(np.diagonal(species)))


def test_theEntriesAreTheRatesDerivatives(my_setup):
    # central differences at a random state, a relative step of 1e-4:
    # measured 1.3e-10 of each species entry and 1.7e-7 of each temperature
    # entry
    rng = np.random.default_rng(3)
    mb = cell(mechanism, "explicit", 10e5, 1800.0, np.ones(12) / 12)
    p, T, Y = randomState(mb.simulator.mixture, rng)
    mb = cell(mechanism, "explicit", p, T, Y, solver=withJacobian)
    J = jacobian(mb)
    assertOnlyTheRung(J)
    diagonal = bySpeciesDifferences(mb, 1e-4 * at(mb, "Q")[5:], (1.0, -1.0))
    assert np.abs(np.diagonal(J[:, 5:]) / diagonal - 1.0).max() < 1e-8
    byT = byTemperatureDifferences(mb, 1e-4 * T)
    assert np.abs(J[:, 4] / byT - 1.0).max() < 1e-5


def test_theEntriesAreFiniteAtTheFloor(my_setup):
    # the fresh stoichiometric mixture at 20 bar and 1500 K, a tenth CO2 so
    # the last species has room to take up an increase: the radicals are
    # none, so their rows are the floor's limit, forward differences from
    # zero with an absolute step of 1e-8 of the density; measured 8.7e-8
    # of each species entry, the difference's own truncation (8.7e-7 at
    # 1e-7). The temperature entries by central differences at 1e-5 of T:
    # measured 2.9e-8, again the truncation (2.5e-6 at 1e-4)
    Y = 0.9 * stoichiometric()
    Y[-1] = 0.1
    mb = cell(mechanism, "explicit", 20e5, 1500.0, Y, solver=withJacobian)
    J = jacobian(mb)
    assert np.isfinite(J).all()
    Q = at(mb, "Q")
    diagonal = bySpeciesDifferences(
        mb, np.maximum(1e-4 * Q[5:], 1e-8 * Q[0]), (1.0, 0.0)
    )
    assert np.abs(np.diagonal(J[:, 5:]) / diagonal - 1.0).max() < 1e-6
    byT = byTemperatureDifferences(mb, 1e-5 * 1500.0)
    assert np.abs(J[:, 4] / byT - 1.0).max() < 1e-6
    single = cell(
        mechanism, "explicit", 20e5, 1500.0, Y, precision="single", solver=withJacobian
    )
    assert np.isfinite(jacobian(single)).all()
