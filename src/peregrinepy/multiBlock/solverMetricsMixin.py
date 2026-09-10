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

from .metricsMixin import MetricsMixin


class SolverMetricsMixin(MetricsMixin):
    @staticmethod
    def _corner(coords, axes, *highs):
        """The (x, y, z) of one corner of every face plane, stacked. :axes:
        are the two the plane spans, :highs: which end of each to take."""
        s = [slice(None)] * 3
        for n, high in zip(axes, highs):
            s[n] = np.s_[1:] if high else np.s_[:-1]
        return np.stack([c[tuple(s)] for c in coords], axis=-1)

    def computeMetrics(self):
        """The cell centers every block has, and then everything a flux needs:
        the face vectors it is taken through, the cell volumes it is divided
        by, and the transformation metrics the diffusion operator
        differentiates through."""
        super().computeMetrics()

        x = self.array["x"]
        y = self.array["y"]
        z = self.array["z"]
        coords = (x, y, z)

        # ----------------------------------------------------------------------------
        # Face centers, area vectors and normals
        # ----------------------------------------------------------------------------
        # The three are one computation with the axis rotated: a face center is
        # the mean of its four corners, and its area vector is half the cross
        # product of the quad's diagonals, taken in the cyclic axis pair so the
        # normal points out of the low face.
        for a, axis in enumerate("ijk"):
            inPlane = [n for n in range(3) if n != a]
            diagonal = [(a + 1) % 3, (a + 2) % 3]

            center = 0.25 * (
                self._corner(coords, inPlane, 0, 0)
                + self._corner(coords, inPlane, 0, 1)
                + self._corner(coords, inPlane, 1, 0)
                + self._corner(coords, inPlane, 1, 1)
            )
            for n, c in enumerate("xyz"):
                self.array[f"{axis}{c}c"][:] = center[..., n]

            S = 0.5 * np.cross(
                self._corner(coords, diagonal, 1, 0)
                - self._corner(coords, diagonal, 0, 1),
                self._corner(coords, diagonal, 1, 1)
                - self._corner(coords, diagonal, 0, 0),
            )
            for n, c in enumerate("xyz"):
                self.array[f"{axis}s{c}"][:] = S[..., n]

            area = self.array[f"{axis}S"]
            area[:] = np.sqrt(
                self.array[f"{axis}sx"] ** 2
                + self.array[f"{axis}sy"] ** 2
                + self.array[f"{axis}sz"] ** 2
            )
            np.clip(area, 1e-16, None, out=area)

            for c in "xyz":
                self.array[f"{axis}n{c}"][:] = self.array[f"{axis}s{c}"] / area

            self.updateDeviceView(
                [f"{axis}{c}c" for c in "xyz"]
                + [f"{axis}s{c}" for c in "xyz"]
                + [f"{axis}S"]
                + [f"{axis}n{c}" for c in "xyz"]
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

        for a, (axis, length) in enumerate(zip("ijk", ("dI", "dJ", "dK"))):
            far, near = [slice(None)] * 3, [slice(None)] * 3
            far[a], near[a] = np.s_[1:], np.s_[:-1]
            far, near = tuple(far), tuple(near)
            self.array[length][:] = np.sqrt(
                sum(
                    (self.array[f"{axis}{c}c"][far] - self.array[f"{axis}{c}c"][near])
                    ** 2
                    for c in "xyz"
                )
            )

        self.updateDeviceView(["dI", "dJ", "dK"])

        # ----------------------------------------------------------------------------
        # Cell center transformation metrics (ferda FD diffusion operator)
        # second order only
        # ----------------------------------------------------------------------------

        # the eight cell corners, numbered as the diagram above
        c1, c2, c3, c4, c5, c6, c7, c8 = (
            self._corner(coords, (0, 1, 2), *highs)
            for highs in (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
                (1, 0, 1),
            )
        )

        # Derivative of (x,y,z) w.r.t. (E,N,C), each the mean of the four
        # edges running that way
        dE = 0.25 * ((c4 - c1) + (c8 - c5) + (c3 - c2) + (c7 - c6))
        dN = 0.25 * ((c2 - c1) + (c3 - c4) + (c7 - c8) + (c6 - c5))
        dC = 0.25 * ((c5 - c1) + (c8 - c4) + (c6 - c2) + (c7 - c3))

        # the inverse of that jacobian is its adjugate over its determinant,
        # and the adjugate's rows are the cross products of the other two
        J = self.array["J"]
        for name, rows in (
            ("dEd", np.cross(dN, dC)),
            ("dNd", np.cross(dC, dE)),
            ("dCd", np.cross(dE, dN)),
        ):
            for n, comp in enumerate("xyz"):
                self.array[f"{name}{comp}"][:] = rows[..., n] / J

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
