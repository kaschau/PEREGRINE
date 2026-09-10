import numpy as np

from .baseMesher import BaseMesher


class AnnulusMesher(BaseMesher):
    """A wedge of an annulus swept about the axis p1->p2, split into a lattice
    of wedges. p3 fixes where the sweep starts and how far the inner radius
    stands off the axis, and the wedge reaches :thickness: further out."""

    mesherName = "annulus"

    def __init__(
        self,
        p1=[0, 0, 0],
        p2=[1, 0, 0],
        p3=[0, 1, 0],
        sweep=45,
        thickness=0.1,
        periodic=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.p3 = np.array(p3, dtype=np.float64)
        self.sweep = float(sweep)
        self.thickness = float(thickness)
        self.periodic = periodic

        length = np.linalg.norm(self.p2 - self.p1)
        innerRadius = np.linalg.norm(self.p3 - self.p1)
        if length == 0.0:
            raise ValueError("p1 and p2 must differ, the line between them is the axis")
        if innerRadius == 0.0:
            raise ValueError(
                "p3 must be off the axis, its distance is the inner radius"
            )

        self.axis = (self.p2 - self.p1) / length
        self.radial = (self.p3 - self.p1) / innerRadius

        if abs(np.dot(self.axis, self.radial)) > 1e-7:
            raise ValueError("the line (p1,p3) is not orthogonal to (p1,p2)")
        # sweeping backwards turns every cell inside out; p2 is what you flip
        # to go the other way round
        if not 0.0 < self.sweep <= 360.0:
            raise ValueError(
                f"sweep must be greater than 0 and no more than 360, not {sweep}"
            )
        # a rotational periodic turns a face about the axis and translates it
        # nowhere, so a wedge that does not close on itself has to be swept
        # about an axis through the origin for its faces to land on each other
        if self.periodic and self.sweep != 360.0:
            offAxis = self.p1 - np.dot(self.p1, self.axis) * self.axis
            if np.linalg.norm(offAxis) > 1e-7:
                raise ValueError(
                    "a rotationally periodic annulus must be swept about an axis "
                    "through the origin"
                )

        # where each block of the lattice starts and ends, along the axis, out
        # from it, and around it
        self.edges = [
            np.linspace(0.0, length, self.mbDims[0] + 1),
            np.linspace(innerRadius, innerRadius + self.thickness, self.mbDims[1] + 1),
            np.linspace(0.0, self.sweep, self.mbDims[2] + 1),
        ]

    @property
    def periodicAxes(self):
        return (False, False, self.periodic)

    def shapeBlock(self, blk, i, j, k):
        blk.setExtents(*self.dimsPerBlock)

        x, r, theta = (
            np.linspace(self.edges[n][m], self.edges[n][m + 1], self.dimsPerBlock[n])
            for n, m in enumerate((i, j, k))
        )
        radial = self._rotate(self.radial, np.radians(theta))

        pts = (
            self.p1
            + x[:, None, None, None] * self.axis
            + r[None, :, None, None] * radial[None, None, :, :]
        )

        s_i = blk.interior
        for n, var in enumerate(("x", "y", "z")):
            blk.array[var][s_i] = pts[:, :, :, n]

    def setPeriodicFaces(self, blk, i, j, k):
        """A full turn closes on itself; anything less is a rotational periodic."""
        if not self.periodic:
            return
        for edge, nface in ((0, 5), (self.mbDims[2] - 1, 6)):
            if k != edge:
                continue
            face = blk.getFace(nface)
            if self.sweep == 360.0:
                face.bcType = "interior"
            else:
                # the low face takes its halo from the far end of the sweep, so
                # it is turned back, and the high face the other way
                face.bcType = "periodicRot"
                sweep = -self.sweep if face.amILow else self.sweep
                face.setPeriodic(rotation=face.rotationAbout(self.axis, sweep))

    def _rotate(self, p, theta):
        """p turned about the annulus axis by each angle in :theta:.
        See http://paulbourke.net/geometry/rotate/"""
        n = self.axis
        ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
        return ct * p + st * np.cross(n, p) + (1.0 - ct) * np.dot(n, p) * n
