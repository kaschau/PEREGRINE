import itertools

import numpy as np
import peregrinepy as pg
import pytest


##############################################
# Test all inlet boundary conditions
##############################################

pytestmark = pytest.mark.parametrize(
    "adv,spdata",
    list(
        itertools.product(
            ("secondOrderKEEP", "fourthOrderKEEP"),
            (["Air"], "thtr_CH4_O2_Stanford_Skeletal.yaml"),
        )
    ),
)


class TestPeriodics:
    @classmethod
    def setup_class(self):
        pass

    @classmethod
    def teardown_class(self):
        pass

    def test_rotationalPeriodics(self, my_setup, adv, spdata):

        config = pg.files.configFile()
        config["RHS"]["primaryAdvFlux"] = adv
        config["thermochem"]["spdata"] = spdata

        mb = pg.multiBlock.generateMultiBlockSolver(1, config)

        axis = np.random.random(3)
        axis /= np.linalg.norm(axis)
        sweep = np.random.randint(1, 90)
        p3 = np.random.random(3)
        p3 /= np.linalg.norm(axis)
        p3[0] = (-axis[1] * p3[1] - axis[2] * p3[2]) / axis[0]

        pg.grid.create.multiBlockAnnulus(
            mb,
            sweep=sweep,
            p2=axis,
            p3=p3,
            periodic=True,
        )
        blk = mb[0]
        blk.getFace(5).commRank = 0
        blk.getFace(6).commRank = 0

        mb.setBlockCommunication()
        mb.initSolverArrays(config)

        mb.unifyGrid()
        mb.computeMetrics(config["RHS"]["diffOrder"])

        qshape = blk.array["q"][:, :, :, 0].shape
        # NOTE: Nov, 2021 KAS: The currently un protected extrapolation of
        # boundary conditions were making the constantMassFluxInlet test case
        # behave poorly (negative species, etc.). So instead of random physical
        # values everywhere we narrow the scope a bit. Maybe down the line
        # we see how necessary it is to protect those BC extraplations.
        p = np.random.uniform(low=101325 * 0.9, high=101325 * 1.1)
        u = np.random.uniform(low=1, high=1000, size=qshape)
        v = np.random.uniform(low=1, high=1000, size=qshape)
        w = np.random.uniform(low=1, high=1000, size=qshape)
        T = np.random.uniform(low=300 * 0.9, high=300 * 1.1)

        if blk.ns > 1:
            Y = np.random.uniform(low=0.0, high=1.0, size=(blk.ns - 1))
            Y = Y / np.sum(Y)

        blk.array["q"][:, :, :, 0] = p
        blk.array["q"][:, :, :, 1] = u
        blk.array["q"][:, :, :, 2] = v
        blk.array["q"][:, :, :, 3] = w
        blk.array["q"][:, :, :, 4] = T
        if blk.ns > 1:
            blk.array["q"][:, :, :, 5::] = Y
        blk.updateDeviceView("q")

        mb.eos(blk, mb.thtrdat, 0, "prims")
        pg.consistify(mb)

        blk.updateHostView(["q", "dqdx", "dqdy", "dqdz"])

        u = blk.array["q"][:, :, :, 1]
        v = blk.array["q"][:, :, :, 2]
        w = blk.array["q"][:, :, :, 3]

        dqdx = blk.array["dqdx"]
        dqdy = blk.array["dqdy"]
        dqdz = blk.array["dqdz"]

        nx = blk.array["knx"]
        ny = blk.array["kny"]
        nz = blk.array["knz"]

        ng = blk.ng
        for g in range(ng):

            # face5 halo compares to interior on side 6
            s5 = np.s_[:, :, g]
            s6c = np.s_[:, :, -2 * ng + g]
            s6f = np.s_[:, :, -2 * ng - 1 + g]

            normals5 = np.column_stack((nx[s5].ravel(), ny[s5].ravel(), nz[s5].ravel()))
            velo5 = np.column_stack((u[s5].ravel(), v[s5].ravel(), w[s5].ravel()))

            normals6 = np.column_stack(
                (nx[s6f].ravel(), ny[s6f].ravel(), nz[s6f].ravel())
            )
            velo6 = np.column_stack((u[s6c].ravel(), v[s6c].ravel(), w[s6c].ravel()))

            assert np.allclose(
                np.sum(normals5 * velo5, axis=1), np.sum(normals6 * velo6, axis=1)
            )

            # check the gradients
            for i in range(blk.ne):
                dqdx5 = np.column_stack(
                    (
                        dqdx[s5][:, :, i].ravel(),
                        dqdy[s5][:, :, i].ravel(),
                        dqdz[s5][:, :, i].ravel(),
                    )
                )

                dqdx6 = np.column_stack(
                    (
                        dqdx[s6c][:, :, i].ravel(),
                        dqdy[s6c][:, :, i].ravel(),
                        dqdz[s6c][:, :, i].ravel(),
                    )
                )
                assert np.allclose(
                    np.sum(normals5 * dqdx5, axis=1), np.sum(normals6 * dqdx6, axis=1)
                )

            # Now check the other way
            # face6 halo compares to interior on side 5
            s6 = np.s_[:, :, -(g + 1)]
            s5c = np.s_[:, :, 2 * ng - g - 1]
            s5f = np.s_[:, :, 2 * ng - g]

            normals6 = np.column_stack((nx[s6].ravel(), ny[s6].ravel(), nz[s6].ravel()))
            velo6 = np.column_stack((u[s6].ravel(), v[s6].ravel(), w[s6].ravel()))

            normals5 = np.column_stack(
                (nx[s5f].ravel(), ny[s5f].ravel(), nz[s5f].ravel())
            )
            velo5 = np.column_stack((u[s5c].ravel(), v[s5c].ravel(), w[s5c].ravel()))

            assert np.allclose(
                np.sum(normals6 * velo6, axis=1), np.sum(normals5 * velo5, axis=1)
            )

            # check the gradients
            for i in range(blk.ne):
                dqdx6 = np.column_stack(
                    (
                        dqdx[s6][:, :, i].ravel(),
                        dqdy[s6][:, :, i].ravel(),
                        dqdz[s6][:, :, i].ravel(),
                    )
                )

                dqdx5 = np.column_stack(
                    (
                        dqdx[s5c][:, :, i].ravel(),
                        dqdy[s5c][:, :, i].ravel(),
                        dqdz[s5c][:, :, i].ravel(),
                    )
                )
                assert np.allclose(
                    np.sum(normals6 * dqdx6, axis=1), np.sum(normals5 * dqdx5, axis=1)
                )
