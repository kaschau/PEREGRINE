"""The density's derivatives every eos gives the dual time preconditioning
-- by p, by T, and by each carried mass fraction with the last taking up
the change -- against central differences of the state the same eos makes
from primitives."""

import numpy as np
import pytest

import peregrinepy as pg
from peregrinepy.kernel import CellCenterKernel
from peregrinepy.multiBlock.arrays import CellCenterArray

from ..gases import primitives


class withDerivatives(pg.multiBlock.solver):
    """A solver with the derivatives' array and kernel declared."""

    def _declArrays(self):
        super()._declArrays()
        self.declArray("qj", CellCenterArray, components=self.ne - 3)

    def _declKernels(self):
        super()._declKernels()
        self.kernels["densityDerivatives"] = CellCenterKernel(
            "thermo/densityDerivatives.cpp"
        )


# a calorically perfect gas states its constants; the others fit the library's
mixtures = {
    "cpg": {
        "O2": {"MW": 31.998, "cp0": 918.0},
        "N2": {"MW": 28.014, "cp0": 1040.0},
        "CO2": {"MW": 44.009, "cp0": 844.0},
        "CH4": {"MW": 16.043, "cp0": 2220.0},
    },
    "tpg": ["O2", "N2", "CO2", "CH4"],
    "realGas": ["O2", "N2", "CO2", "CH4"],
}


def case(eos):
    config = pg.files.configFile()
    config["mixture"]["species"] = mixtures[eos]
    config["mixture"]["eos"] = eos
    config["mixture"]["Trange"] = (300.0, 3500.0)
    config["simulation"]["simulator"] = "euler"
    mesh = pg.mesher.CubeMesher(
        mbDims=[1, 1, 1], dimsPerBlock=[2, 2, 2], lengths=[1, 1, 1]
    )
    return withDerivatives(config, mesh)


def density(mb, p, T, Y):
    """rho the eos makes of (p, T, Y) at the one interior cell."""
    blk = mb.blocks[0]
    ng = blk.ng
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5:] = p, 0.0, T, Y[:-1]
    mb.setPrimitives([q])
    return blk.Q.get()[ng, ng, ng, 0]


@pytest.mark.parametrize("eos", ["cpg", "tpg", "realGas"])
def test_derivativesAreTheStatesOwn(my_setup, eos):
    rng = np.random.default_rng(len(eos))
    mb = case(eos)
    blk = mb.blocks[0]
    ng, ns = blk.ng, mb.simulator.mixture.ns
    # a dense cold state, where a real gas is far from ideal
    p, T = 60e5, 320.0
    Y = rng.random(ns)
    Y /= Y.sum()
    rho = density(mb, p, T, Y)
    mb.launch("densityDerivatives", "interior")
    qj = blk.qj.get()[ng, ng, ng]
    # central differences of the state from primitives
    dp, dT = 1e-3 * p, 1e-3 * T
    rho_p = (density(mb, p + dp, T, Y) - density(mb, p - dp, T, Y)) / (2 * dp)
    rho_T = (density(mb, p, T + dT, Y) - density(mb, p, T - dT, Y)) / (2 * dT)
    assert abs(qj[0] / rho_p - 1) < 1e-5 and abs(qj[1] / rho_T - 1) < 1e-5
    for n in range(ns - 1):
        # Y_n up, the last species down by the same
        d = 1e-3 * min(Y[n], Y[-1])
        up, down = Y.copy(), Y.copy()
        up[n], up[-1] = Y[n] + d, Y[-1] - d
        down[n], down[-1] = Y[n] - d, Y[-1] + d
        rho_Y = (density(mb, p, T, up) - density(mb, p, T, down)) / (2 * d)
        assert abs(qj[2 + n] / rho_Y - 1) < 1e-5, (eos, n, qj[2 + n], rho_Y)
    # far from ideal, or the test proved nothing of the cubic (measured 5%)
    if eos == "realGas":
        assert abs(qj[0] * p / rho - 1) > 0.03
