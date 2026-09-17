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
    def faceNormals(self, axis):
        """The area and unit normal of every :axis: face, worked back over
        from the area vector, which is the only one of the three stored."""
        s = getattr(self, f"{axis}S").get()
        # a degenerate face is floored, we divide by this
        area = np.maximum(np.sqrt((s**2).sum(axis=-1)), 1e-16)
        return area, [s[..., n] / area for n in range(3)]

    @staticmethod
    def _corner(nodes, axes, *highs):
        """The (x, y, z) of one corner of every face plane. :axes: are the two
        the plane spans, :highs: which end of each to take."""
        s = [slice(None)] * 3
        for n, high in zip(axes, highs):
            s[n] = np.s_[1:] if high else np.s_[:-1]
        return nodes[tuple(s)]

    def computeMetrics(self):
        """The cell centers every block has, and then everything a flux needs:
        the face vectors it is taken through, the cell volumes it is divided
        by, and the transformation metrics the diffusion operator
        differentiates through."""
        super().computeMetrics()

        nodes = self.nodes.get()
        faces, areas = {}, {}

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
                self._corner(nodes, inPlane, 0, 0)
                + self._corner(nodes, inPlane, 0, 1)
                + self._corner(nodes, inPlane, 1, 0)
                + self._corner(nodes, inPlane, 1, 1)
            )
            faces[axis] = center

            S = 0.5 * np.cross(
                self._corner(nodes, diagonal, 1, 0)
                - self._corner(nodes, diagonal, 0, 1),
                self._corner(nodes, diagonal, 1, 1)
                - self._corner(nodes, diagonal, 0, 0),
            )
            areas[axis] = S
            getattr(self, f"{axis}Faces").set(center)
            getattr(self, f"{axis}S").set(S)

        # ----------------------------------------------------------------------------
        # Cell center volumes
        # ----------------------------------------------------------------------------

        bodyDiagonal = nodes[1::, 1::, 1::] - nodes[0:-1, 0:-1, 0:-1]
        J = (
            sum(
                bodyDiagonal[..., n]
                * (
                    areas["i"][1::, :, :, n]
                    + areas["j"][:, 1::, :, n]
                    + areas["k"][:, :, 1::, n]
                )
                for n in range(3)
            )
            / 3.0e0
        )
        np.clip(J, 1e-16, None, out=J)
        # the kernels only ever divide by it
        self.Jinv.set(1.0 / J)

        # ----------------------------------------------------------------------------
        # Cell lengths, opposite face center to opposite face center
        # ----------------------------------------------------------------------------

        dIJK = np.zeros(self.dIJK.shape)
        for a, axis in enumerate("ijk"):
            far, near = [slice(None)] * 3, [slice(None)] * 3
            far[a], near[a] = np.s_[1:], np.s_[:-1]
            far, near = tuple(far), tuple(near)
            span = faces[axis][far] - faces[axis][near]
            dIJK[..., a] = np.sqrt((span**2).sum(axis=-1))
        self.dIJK.set(dIJK)

        # ----------------------------------------------------------------------------
        # Cell center transformation metrics (ferda FD diffusion operator)
        # second order only
        # ----------------------------------------------------------------------------

        # the eight cell corners, numbered as the diagram above
        c1, c2, c3, c4, c5, c6, c7, c8 = (
            self._corner(nodes, (0, 1, 2), *highs)
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
        self.dENCdxyz.set(
            np.stack([np.cross(dN, dC), np.cross(dC, dE), np.cross(dE, dN)], axis=-2)
            / J[..., None, None]
        )
