r"""
The metrics only a solver needs: the face vectors a flux is taken through,
the cell volumes it is divided by, and the transformation metrics the
diffusion operator differentiates through. A block with no solution on it
works out its cell centers and stops.

# The i,j,k block coordinate directions are \Xi (E), \Eta (N), and \Zeta (C)
#
#                  2  o--------------------------o  3
#                     |\                         |\
#                     | \                        | \
#                     |  \                       |  \
#                     |   \                      |   \
#                     |    \ 6                   |    \
#                     |     o--------------------|---- o 7
#                     |     |                    |     |
#                     |     |                    |     |
#                     |     |                    |     |
#                     |     |                    |     |
#   ^ j,N          1  o-----|--------------------o  4  |
#   |                  \    |                     \    |
#   |                   \   |                      \   |
#   |                    \  |                       \  |
#   o-----> i,E           \ |                        \ |
#    \                     \|                         \|
#     \                     o------------------------- o
#      v  k,C             5                              8
#
"""

import numpy as np


class SolverMetricsMixin:
    def computeSolverMetrics(self):
        """Everything past the cell centers: the face vectors a flux is taken
        through, the cell volumes it is divided by, and the transformation
        metrics the diffusion operator differentiates through."""
        x = self.array["x"]
        y = self.array["y"]
        z = self.array["z"]

        # ----------------------------------------------------------------------------
        # i face centers, area, normal vectors
        # ----------------------------------------------------------------------------
        self.array["ixc"][:] = 0.25 * (
            x[:, 0:-1, 0:-1] + x[:, 0:-1, 1::] + x[:, 1::, 0:-1] + x[:, 1::, 1::]
        )
        self.array["iyc"][:] = 0.25 * (
            y[:, 0:-1, 0:-1] + y[:, 0:-1, 1::] + y[:, 1::, 0:-1] + y[:, 1::, 1::]
        )
        self.array["izc"][:] = 0.25 * (
            z[:, 0:-1, 0:-1] + z[:, 0:-1, 1::] + z[:, 1::, 0:-1] + z[:, 1::, 1::]
        )

        S1265 = np.zeros(list(self.array["isx"][:].shape) + [3])
        vectorX1 = np.zeros(S1265.shape)
        vectorX1[:, :, :, 0] = x[:, 0:-1, 0:-1]
        vectorX1[:, :, :, 1] = y[:, 0:-1, 0:-1]
        vectorX1[:, :, :, 2] = z[:, 0:-1, 0:-1]
        vectorX2 = np.zeros(S1265.shape)
        vectorX2[:, :, :, 0] = x[:, 1::, 0:-1]
        vectorX2[:, :, :, 1] = y[:, 1::, 0:-1]
        vectorX2[:, :, :, 2] = z[:, 1::, 0:-1]
        vectorX5 = np.zeros(S1265.shape)
        vectorX5[:, :, :, 0] = x[:, 0:-1, 1::]
        vectorX5[:, :, :, 1] = y[:, 0:-1, 1::]
        vectorX5[:, :, :, 2] = z[:, 0:-1, 1::]
        vectorX6 = np.zeros(S1265.shape)
        vectorX6[:, :, :, 0] = x[:, 1::, 1::]
        vectorX6[:, :, :, 1] = y[:, 1::, 1::]
        vectorX6[:, :, :, 2] = z[:, 1::, 1::]
        S1265 = 0.5 * np.cross(vectorX2 - vectorX5, vectorX6 - vectorX1)
        self.array["isx"][:] = np.dot(S1265, np.array([1.0, 0.0, 0.0]))
        self.array["isy"][:] = np.dot(S1265, np.array([0.0, 1.0, 0.0]))
        self.array["isz"][:] = np.dot(S1265, np.array([0.0, 0.0, 1.0]))

        self.array["iS"][:] = np.sqrt(
            self.array["isx"] ** 2 + self.array["isy"] ** 2 + self.array["isz"] ** 2
        )

        np.clip(self.array["iS"], 1e-16, None, out=self.array["iS"])

        self.array["inx"][:] = self.array["isx"] / self.array["iS"]
        self.array["iny"][:] = self.array["isy"] / self.array["iS"]
        self.array["inz"][:] = self.array["isz"] / self.array["iS"]

        self.updateDeviceView(
            [
                "ixc",
                "iyc",
                "izc",
                "isx",
                "isy",
                "isz",
                "iS",
                "inx",
                "iny",
                "inz",
            ]
        )

        # ----------------------------------------------------------------------------
        # j face center, area, normal vectors
        # ----------------------------------------------------------------------------
        self.array["jxc"][:] = 0.25 * (
            x[0:-1, :, 0:-1] + x[0:-1, :, 1::] + x[1::, :, 0:-1] + x[1::, :, 1::]
        )
        self.array["jyc"][:] = 0.25 * (
            y[0:-1, :, 0:-1] + y[0:-1, :, 1::] + y[1::, :, 0:-1] + y[1::, :, 1::]
        )
        self.array["jzc"][:] = 0.25 * (
            z[0:-1, :, 0:-1] + z[0:-1, :, 1::] + z[1::, :, 0:-1] + z[1::, :, 1::]
        )

        S1584 = np.zeros(list(self.array["jsx"][:].shape) + [3])
        vectorX1 = np.zeros(S1584.shape)
        vectorX1[:, :, :, 0] = x[0:-1, :, 0:-1]
        vectorX1[:, :, :, 1] = y[0:-1, :, 0:-1]
        vectorX1[:, :, :, 2] = z[0:-1, :, 0:-1]
        vectorX4 = np.zeros(S1584.shape)
        vectorX4[:, :, :, 0] = x[1::, :, 0:-1]
        vectorX4[:, :, :, 1] = y[1::, :, 0:-1]
        vectorX4[:, :, :, 2] = z[1::, :, 0:-1]
        vectorX5 = np.zeros(S1584.shape)
        vectorX5[:, :, :, 0] = x[0:-1, :, 1::]
        vectorX5[:, :, :, 1] = y[0:-1, :, 1::]
        vectorX5[:, :, :, 2] = z[0:-1, :, 1::]
        vectorX8 = np.zeros(S1584.shape)
        vectorX8[:, :, :, 0] = x[1::, :, 1::]
        vectorX8[:, :, :, 1] = y[1::, :, 1::]
        vectorX8[:, :, :, 2] = z[1::, :, 1::]
        S1584 = 0.5 * np.cross(vectorX5 - vectorX4, vectorX8 - vectorX1)
        self.array["jsx"][:] = np.dot(S1584, np.array([1.0, 0.0, 0.0]))
        self.array["jsy"][:] = np.dot(S1584, np.array([0.0, 1.0, 0.0]))
        self.array["jsz"][:] = np.dot(S1584, np.array([0.0, 0.0, 1.0]))

        self.array["jS"][:] = np.sqrt(
            self.array["jsx"] ** 2 + self.array["jsy"] ** 2 + self.array["jsz"] ** 2
        )

        np.clip(self.array["jS"], 1e-16, None, out=self.array["jS"])

        self.array["jnx"][:] = self.array["jsx"] / self.array["jS"]
        self.array["jny"][:] = self.array["jsy"] / self.array["jS"]
        self.array["jnz"][:] = self.array["jsz"] / self.array["jS"]

        self.updateDeviceView(
            [
                "jxc",
                "jyc",
                "jzc",
                "jsx",
                "jsy",
                "jsz",
                "jS",
                "jnx",
                "jny",
                "jnz",
            ]
        )

        # ----------------------------------------------------------------------------
        # k face center, area, normal vectors
        # ----------------------------------------------------------------------------
        self.array["kxc"][:] = 0.25 * (
            x[0:-1, 0:-1, :] + x[0:-1, 1::, :] + x[1::, 0:-1, :] + x[1::, 1::, :]
        )
        self.array["kyc"][:] = 0.25 * (
            y[0:-1, 0:-1, :] + y[0:-1, 1::, :] + y[1::, 0:-1, :] + y[1::, 1::, :]
        )
        self.array["kzc"][:] = 0.25 * (
            z[0:-1, 0:-1, :] + z[0:-1, 1::, :] + z[1::, 0:-1, :] + z[1::, 1::, :]
        )

        S1432 = np.zeros(list(self.array["ksx"][:].shape) + [3])
        vectorX1 = np.zeros(S1432.shape)
        vectorX1[:, :, :, 0] = x[0:-1, 0:-1, :]
        vectorX1[:, :, :, 1] = y[0:-1, 0:-1, :]
        vectorX1[:, :, :, 2] = z[0:-1, 0:-1, :]
        vectorX2 = np.zeros(S1432.shape)
        vectorX2[:, :, :, 0] = x[0:-1, 1::, :]
        vectorX2[:, :, :, 1] = y[0:-1, 1::, :]
        vectorX2[:, :, :, 2] = z[0:-1, 1::, :]
        vectorX3 = np.zeros(S1432.shape)
        vectorX3[:, :, :, 0] = x[1::, 1::, :]
        vectorX3[:, :, :, 1] = y[1::, 1::, :]
        vectorX3[:, :, :, 2] = z[1::, 1::, :]
        vectorX4 = np.zeros(S1432.shape)
        vectorX4[:, :, :, 0] = x[1::, 0:-1, :]
        vectorX4[:, :, :, 1] = y[1::, 0:-1, :]
        vectorX4[:, :, :, 2] = z[1::, 0:-1, :]
        S1432 = 0.5 * np.cross(vectorX4 - vectorX2, vectorX3 - vectorX1)
        self.array["ksx"][:] = np.dot(S1432, np.array([1.0, 0.0, 0.0]))
        self.array["ksy"][:] = np.dot(S1432, np.array([0.0, 1.0, 0.0]))
        self.array["ksz"][:] = np.dot(S1432, np.array([0.0, 0.0, 1.0]))

        self.array["kS"][:] = np.sqrt(
            self.array["ksx"] ** 2 + self.array["ksy"] ** 2 + self.array["ksz"] ** 2
        )

        np.clip(self.array["kS"], 1e-16, None, out=self.array["kS"])

        self.array["knx"][:] = self.array["ksx"] / self.array["kS"]
        self.array["kny"][:] = self.array["ksy"] / self.array["kS"]
        self.array["knz"][:] = self.array["ksz"] / self.array["kS"]

        self.updateDeviceView(
            [
                "kxc",
                "kyc",
                "kzc",
                "ksx",
                "ksy",
                "ksz",
                "kS",
                "knx",
                "kny",
                "knz",
            ]
        )

        # ----------------------------------------------------------------------------
        # Cell center volumes
        # ----------------------------------------------------------------------------

        self.array["J"][:] = (
            (x[1::, 1::, 1::] - x[0:-1, 0:-1, 0:-1])
            * (
                self.array["isx"][1::, :, :]
                + self.array["jsx"][:, 1::, :]
                + self.array["ksx"][:, :, 1::]
            )
            + (y[1::, 1::, 1::] - y[0:-1, 0:-1, 0:-1])
            * (
                self.array["isy"][1::, :, :]
                + self.array["jsy"][:, 1::, :]
                + self.array["ksy"][:, :, 1::]
            )
            + (z[1::, 1::, 1::] - z[0:-1, 0:-1, 0:-1])
            * (
                self.array["isz"][1::, :, :]
                + self.array["jsz"][:, 1::, :]
                + self.array["ksz"][:, :, 1::]
            )
        ) / 3.0e0

        np.clip(self.array["J"], 1e-16, None, out=self.array["J"])

        self.updateDeviceView(["J"])

        # ----------------------------------------------------------------------------
        # Cell lengths, opposite face center to opposite face center
        # ----------------------------------------------------------------------------

        self.array["dI"][:] = np.sqrt(
            (self.array["ixc"][1::, :, :] - self.array["ixc"][0:-1, :, :]) ** 2
            + (self.array["iyc"][1::, :, :] - self.array["iyc"][0:-1, :, :]) ** 2
            + (self.array["izc"][1::, :, :] - self.array["izc"][0:-1, :, :]) ** 2
        )
        self.array["dJ"][:] = np.sqrt(
            (self.array["jxc"][:, 1::, :] - self.array["jxc"][:, 0:-1, :]) ** 2
            + (self.array["jyc"][:, 1::, :] - self.array["jyc"][:, 0:-1, :]) ** 2
            + (self.array["jzc"][:, 1::, :] - self.array["jzc"][:, 0:-1, :]) ** 2
        )
        self.array["dK"][:] = np.sqrt(
            (self.array["kxc"][:, :, 1::] - self.array["kxc"][:, :, 0:-1]) ** 2
            + (self.array["kyc"][:, :, 1::] - self.array["kyc"][:, :, 0:-1]) ** 2
            + (self.array["kzc"][:, :, 1::] - self.array["kzc"][:, :, 0:-1]) ** 2
        )

        self.updateDeviceView(["dI", "dJ", "dK"])

        # ----------------------------------------------------------------------------
        # Cell center transformation metrics (ferda FD diffusion operator)
        # second order only
        # ----------------------------------------------------------------------------

        # cell corners
        x1 = x[0:-1, 0:-1, 0:-1]
        x2 = x[0:-1, 1::, 0:-1]
        x3 = x[1::, 1::, 0:-1]
        x4 = x[1::, 0:-1, 0:-1]
        x5 = x[0:-1, 0:-1, 1::]
        x6 = x[0:-1, 1::, 1::]
        x7 = x[1::, 1::, 1::]
        x8 = x[1::, 0:-1, 1::]

        y1 = y[0:-1, 0:-1, 0:-1]
        y2 = y[0:-1, 1::, 0:-1]
        y3 = y[1::, 1::, 0:-1]
        y4 = y[1::, 0:-1, 0:-1]
        y5 = y[0:-1, 0:-1, 1::]
        y6 = y[0:-1, 1::, 1::]
        y7 = y[1::, 1::, 1::]
        y8 = y[1::, 0:-1, 1::]

        z1 = z[0:-1, 0:-1, 0:-1]
        z2 = z[0:-1, 1::, 0:-1]
        z3 = z[1::, 1::, 0:-1]
        z4 = z[1::, 0:-1, 0:-1]
        z5 = z[0:-1, 0:-1, 1::]
        z6 = z[0:-1, 1::, 1::]
        z7 = z[1::, 1::, 1::]
        z8 = z[1::, 0:-1, 1::]

        # Derivative of (x,y,z) w.r.t. (E,N,C)
        dxdE = 0.25 * ((x4 - x1) + (x8 - x5) + (x3 - x2) + (x7 - x6))
        dydE = 0.25 * ((y4 - y1) + (y8 - y5) + (y3 - y2) + (y7 - y6))
        dzdE = 0.25 * ((z4 - z1) + (z8 - z5) + (z3 - z2) + (z7 - z6))

        dxdN = 0.25 * ((x2 - x1) + (x3 - x4) + (x7 - x8) + (x6 - x5))
        dydN = 0.25 * ((y2 - y1) + (y3 - y4) + (y7 - y8) + (y6 - y5))
        dzdN = 0.25 * ((z2 - z1) + (z3 - z4) + (z7 - z8) + (z6 - z5))

        dxdC = 0.25 * ((x5 - x1) + (x8 - x4) + (x6 - x2) + (x7 - x3))
        dydC = 0.25 * ((y5 - y1) + (y8 - y4) + (y6 - y2) + (y7 - y3))
        dzdC = 0.25 * ((z5 - z1) + (z8 - z4) + (z6 - z2) + (z7 - z3))

        self.array["dEdx"][:] = (dydN * dzdC - dydC * dzdN) / self.array["J"]
        self.array["dEdy"][:] = (dxdN * dzdC - dxdC * dzdN) / -self.array["J"]
        self.array["dEdz"][:] = (dxdN * dydC - dxdC * dydN) / self.array["J"]

        self.array["dNdx"][:] = (dydE * dzdC - dydC * dzdE) / -self.array["J"]
        self.array["dNdy"][:] = (dxdE * dzdC - dxdC * dzdE) / self.array["J"]
        self.array["dNdz"][:] = (dxdE * dydC - dxdC * dydE) / -self.array["J"]

        self.array["dCdx"][:] = (dydE * dzdN - dydN * dzdE) / self.array["J"]
        self.array["dCdy"][:] = (dxdE * dzdN - dxdN * dzdE) / -self.array["J"]
        self.array["dCdz"][:] = (dxdE * dydN - dxdN * dydE) / self.array["J"]

        self.updateDeviceView(
            [
                "dEdx",
                "dEdy",
                "dEdz",
                "dNdx",
                "dNdy",
                "dNdz",
                "dCdx",
                "dCdy",
                "dCdz",
            ]
        )

        # # fourth order (not used)
        # xc = self.array["xc"]
        # yc = self.array["yc"]
        # zc = self.array["zc"]

        # dxdE = (
        #     -xc[4::, 2:-2, 2:-2]
        #     + 8.0 * xc[3:-1, 2:-2, 2:-2]
        #     - 8.0 * xc[1:-3, 2:-2, 2:-2]
        #     + xc[0:-4, 2:-2, 2:-2]
        # ) / 12.0
        # dxdN = (
        #     -xc[2:-2, 4::, 2:-2]
        #     + 8.0 * xc[2:-2, 3:-1, 2:-2]
        #     - 8.0 * xc[2:-2, 1:-3, 2:-2]
        #     + xc[2:-2, 0:-4, 2:-2]
        # ) / 12.0
        # dxdC = (
        #     -xc[2:-2, 2:-2, 4::]
        #     + 8.0 * xc[2:-2, 2:-2, 3:-1]
        #     - 8.0 * xc[2:-2, 2:-2, 1:-3]
        #     + xc[2:-2, 2:-2, 0:-4]
        # ) / 12.0

        # dydE = (
        #     -yc[4::, 2:-2, 2:-2]
        #     + 8.0 * yc[3:-1, 2:-2, 2:-2]
        #     - 8.0 * yc[1:-3, 2:-2, 2:-2]
        #     + yc[0:-4, 2:-2, 2:-2]
        # ) / 12.0
        # dydN = (
        #     -yc[2:-2, 4::, 2:-2]
        #     + 8.0 * yc[2:-2, 3:-1, 2:-2]
        #     - 8.0 * yc[2:-2, 1:-3, 2:-2]
        #     + yc[2:-2, 0:-4, 2:-2]
        # ) / 12.0
        # dydC = (
        #     -yc[2:-2, 2:-2, 4::]
        #     + 8.0 * yc[2:-2, 2:-2, 3:-1]
        #     - 8.0 * yc[2:-2, 2:-2, 1:-3]
        #     + yc[2:-2, 2:-2, 0:-4]
        # ) / 12.0

        # dzdE = (
        #     -zc[4::, 2:-2, 2:-2]
        #     + 8.0 * zc[3:-1, 2:-2, 2:-2]
        #     - 8.0 * zc[1:-3, 2:-2, 2:-2]
        #     + zc[0:-4, 2:-2, 2:-2]
        # ) / 12.0
        # dzdN = (
        #     -zc[2:-2, 4::, 2:-2]
        #     + 8.0 * zc[2:-2, 3:-1, 2:-2]
        #     - 8.0 * zc[2:-2, 1:-3, 2:-2]
        #     + zc[2:-2, 0:-4, 2:-2]
        # ) / 12.0
        # dzdC = (
        #     -zc[2:-2, 2:-2, 4::]
        #     + 8.0 * zc[2:-2, 2:-2, 3:-1]
        #     - 8.0 * zc[2:-2, 2:-2, 1:-3]
        #     + zc[2:-2, 2:-2, 0:-4]
        # ) / 12.0

        # self.array["dEdx"][2:-2, 2:-2, 2:-2] = (dydN * dzdC - dydC * dzdN) / self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dEdy"][2:-2, 2:-2, 2:-2] = (dxdN * dzdC - dxdC * dzdN) / -self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dEdz"][2:-2, 2:-2, 2:-2] = (dxdN * dydC - dxdC * dydN) / self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]

        # self.array["dNdx"][2:-2, 2:-2, 2:-2] = (dydE * dzdC - dydC * dzdE) / -self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dNdy"][2:-2, 2:-2, 2:-2] = (dxdE * dzdC - dxdC * dzdE) / self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dNdz"][2:-2, 2:-2, 2:-2] = (dxdE * dydC - dxdC * dydE) / -self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]

        # self.array["dCdx"][2:-2, 2:-2, 2:-2] = (dydE * dzdN - dydN * dzdE) / self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dCdy"][2:-2, 2:-2, 2:-2] = (dxdE * dzdN - dxdN * dzdE) / -self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
        # self.array["dCdz"][2:-2, 2:-2, 2:-2] = (dxdE * dydN - dxdN * dydE) / self.array[
        #     "J"
        # ][2:-2, 2:-2, 2:-2]
