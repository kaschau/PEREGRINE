#!/usr/bin/env python
"""

Test case from

Yuichi Kuya, Soshi Kawai,
High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids,
Journal of Computational Physics,
Volume 442,
2021,
110482,
ISSN 0021-9991,
https://doi.org/10.1016/j.jcp.2021.110482.
(https://www.sciencedirect.com/science/article/pii/S0021999121003776)

Will reproduce test case from section 6.2

"""

from mpi4py import MPI  # noqa: F401

import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


np.seterr(all="raise")


class SkewedCube(pg.mesher.CubeMesher):
    """A periodic cube whose nodes are moved off their lines by a product
    of sines, so no cell is a box."""

    def shapeBlock(self, blk, i, j, k):
        super().shapeBlock(blk, i, j, k)
        NE, NN, NX = self.dimsPerBlock
        xMin = yMin = zMin = -np.pi
        Lx = Ly = Lz = 2 * np.pi
        lamXY = lamYZ = lamXZ = 8.0
        kappa = 0.25
        Ax = Ay = Az = 1.0
        delX, delY, delZ = Lx / (NE - 1), Ly / (NN - 1), Lz / (NX - 1)
        E, N, X = np.meshgrid(
            np.arange(NE), np.arange(NN), np.arange(NX), indexing="ij"
        )
        skew = np.sin(2 * np.pi * kappa)
        nodes = blk.nodes.get()
        skewed = nodes[blk.interior]
        skewed[..., 0] = xMin + delX * (
            E
            + Ax
            * skew
            * np.sin(lamXY * np.pi * N * delY / Ly)
            * np.sin(lamXZ * np.pi * X * delZ / Lz)
        )
        skewed[..., 1] = yMin + delY * (
            N
            + Ay
            * skew
            * np.sin(lamXY * np.pi * E * delX / Lx)
            * np.sin(lamYZ * np.pi * X * delZ / Lz)
        )
        skewed[..., 2] = zMin + delZ * (
            X
            + Az
            * skew
            * np.sin(lamXZ * np.pi * E * delX / Lx)
            * np.sin(lamYZ * np.pi * N * delY / Ly)
        )
        blk.nodes.set(nodes)


def simulate():
    config = pg.files.configFile()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = air
    config.validateConfig()

    NE = 64
    NN = 64
    NX = 64
    mesh = SkewedCube(
        mbDims=[1, 1, 1],
        dimsPerBlock=[NE, NN, NX],
        lengths=[2 * np.pi for _ in range(3)],
        periodic=[True, True, True],
    )
    mb = pg.multiBlock.solver(config, mesh)
    blk = mb.blocks[0]

    R = 287.002507
    cp = 1000.0
    M0 = 0.4
    rho0 = 1.0
    gamma = cp / (cp - R)

    xc, yc, zc = (blk.cells.get()[..., n] for n in range(3))
    # the primitive vector, p, u, v, w, T
    q = np.zeros(blk.Q.shape[:3] + (mb.ne,))
    q[:, :, :, 0] = 1 / gamma + (rho0 * M0**2 / 16.0) * (
        np.cos(2 * xc) + np.cos(2 * yc)
    ) * (np.cos(2 * zc) + 2.0)
    q[:, :, :, 1] = M0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    q[:, :, :, 2] = -M0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    q[:, :, :, 4] = q[:, :, :, 0] / (R * rho0)
    mb.setPrimitives([q])

    dt = 0.1 * 2 * np.pi / NE
    ke = []
    s = []
    t = []
    tEnd = 120 / M0
    bar = pg.misc.Progress(tEnd)
    while mb.tme < tEnd:
        if mb.nrt % 50 == 0:
            bar.at(mb.tme)
            data = mb.exportData(blk, ["rho", "p", "u", "v", "w"])
            rho, p, u, v, w = (data[n][blk.interior] for n in "rho p u v w".split())
            J = 1.0 / blk.Jinv.get()[blk.interior]

            rke = np.sum(0.5 * rho * (u**2 + v**2 + w**2) * J)
            rS = np.sum(rho * np.log10(p * rho ** (-gamma)) * J)

            ke.append(rke)
            s.append(rS)
            t.append(mb.tme * M0)

        mb.integrator.step(dt)

    plt.plot(t, ke / ke[0])
    plt.ylim([0, 2.4])
    plt.title(r"$\rho k / (\rho k)_{0}$")
    plt.savefig("ke.png")
    plt.clf()
    plt.plot(t, (-(s - s[0])) / s[0])
    plt.ylim([-3e-2, 1e-2])
    plt.title(r"$\Delta(\rho s) / (\rho_0 s_0)$")
    plt.savefig("entropy.png")
    plt.clf()


if __name__ == "__main__":
    try:
        pg.backend.abi.lib.initialize()
        simulate()
        pg.backend.abi.lib.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
