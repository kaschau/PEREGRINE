import numpy as np

from .baseMesher import BaseMesher


class AnnulusMesher(BaseMesher):
    """A wedge of an annulus swept about the axis p1->p2, split into a lattice
    of wedges. p3 fixes where the sweep starts."""

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
        self.sweep = sweep
        self.thickness = thickness
        self.periodic = periodic

        # the axis swept about, and how far one block spans along each of them
        self.axis = (self.p2 - self.p1) / np.linalg.norm(self.p2 - self.p1)
        self.dx = np.linalg.norm(self.p2 - self.p1) / self.mbDims[0]
        self.dr = thickness / self.mbDims[1]
        self.dtheta = sweep / self.mbDims[2]

    @property
    def periodicAxes(self):
        return (False, False, self.periodic)

    def startLayer(self, k):
        """Rotate p3 to where this layer of the sweep begins."""
        theta = k * self.dtheta * np.pi / 180.0
        self.layerStart = self._rotate(self.p3 - self.p1, theta) + self.p1
        self.radial = (self.layerStart - self.p1) / np.linalg.norm(
            self.layerStart - self.p1
        )

    def _rotate(self, p, theta):
        """p turned about the annulus axis. See http://paulbourke.net/geometry/rotate/"""
        n, ct, st = self.axis, np.cos(theta), np.sin(theta)
        return np.array(
            [
                (ct + (1 - ct) * n[0] * n[0]) * p[0]
                + ((1 - ct) * n[0] * n[1] - n[2] * st) * p[1]
                + ((1 - ct) * n[0] * n[2] + n[1] * st) * p[2],
                ((1 - ct) * n[0] * n[1] + n[2] * st) * p[0]
                + (ct + (1 - ct) * n[1] * n[1]) * p[1]
                + ((1 - ct) * n[1] * n[2] - n[0] * st) * p[2],
                ((1 - ct) * n[0] * n[2] - n[1] * st) * p[0]
                + ((1 - ct) * n[1] * n[2] + n[0] * st) * p[1]
                + (ct + (1 - ct) * n[2] * n[2]) * p[2],
            ]
        )

    def shapeBlock(self, blk, i, j, k):
        newp1 = self.p1 + self.dx * i * self.axis
        newp2 = self.p1 + self.dx * (i + 1) * self.axis
        newp3 = self.layerStart + self.dx * i * self.axis + self.dr * j * self.radial
        self._annulus(blk, newp1, newp2, newp3, self.dtheta, self.dr, self.dimsPerBlock)

    def setPeriodicFaces(self, blk, i, j, k):
        """A full turn closes on itself; anything less is a rotational periodic."""
        if not self.periodic:
            return
        for edge, nface in ((0, 5), (self.mbDims[2] - 1, 6)):
            if k != edge:
                continue
            face = blk.getFace(nface)
            if float(self.sweep) == 360.0:
                face.bcType = "interior"
            else:
                face.bcType = "periodicRotLow" if nface == 5 else "periodicRotHigh"
                face.bcFam = "periodic"
                face.periodicSpan = self.sweep
                face.periodicAxis = self.axis

    def _annulus(self, blk, p1, p2, p3, sweep, thickness, dimensions):
        """Function to populate the coordinate arrays of a provided peregrinepy.multiBlock.gridBlock in the shape of an annulus with prescribed location, extents, and discretization.
        If the input multiBlock object is a restart block the shape and size of the flow data arrays are also updated.

        Parameters
        ----------

        blk : peregrinepy.blocks.grid_block (or one of its descendants)

        p1 : list, tuple
           List/tuple of length 3 containing the location of the origin of the annulus to be created, i.e.
           the center of the beginning of the cylindrical segment

        p2 : list, tuple
           List/tuple of length 3 containing the location of the end of the annulus to be created, i.e.
           the center of the end of the cylindrical segment

        p3 : list, tuple
           List/tuple of length 3 containing the location of a point orthogonal to the line (p1,p2) marking
           the inner most corner point of the cylindrical segment. This point also serves as the inner radius
           of the cylindrical segment. The outer radius is measured  from :p3: outward along the line (p1,p3)
           a distance of :thickness:. This point also serves as the starting angular point for :sweep: to be measured
           according to the right hand rule about the line (p1,p2). I.e. the variable :sweep: measures the angle
           about which the cylindrical segment "sweeps" starting from the line (p1,p3).

        sweep : float
           Float denoting the angle (in degrees) of sweep of the annular segment in the direction according to the right
           hand rule about the line (p1,p2) starting using the line (p1,p3) as the starting point for the sweep.

        thickness : float
           Float denoting (outer radius - inner radius) of the annulus, where the inner radius is determined by
           the length of the line (p1,p3).

        dimensions : list, tuple
           List/tuple of length 3 containing discretization (ni,nj,nk) in each dimension of the cube. Where the "x"
           direction is along the annulus axis, the "y" direction is along the radial direction, and the "z" direction
           is along the theta direction.

        Returns
        -------
        None
            Updates attributes of parameter blk.

        """

        p1 = np.array([0.0, 0.0, 0.0])  # All periodic axes go through origin!!!
        p2 = np.array(p2)
        p3 = np.array(p3)

        if np.abs(np.dot(p2 - p1, p3 - p1)) > 1e-7:
            raise ValueError("Error: The line (p1,p2) is not orthogonal to (p1,p3)")

        if abs(sweep) < -360 or abs(sweep) > 360.0:
            raise ValueError("Error: sweep parameter must be >-360 and <360")

        n12 = (p2 - p1) / np.linalg.norm(p2 - p1)
        n13 = (p3 - p1) / np.linalg.norm(p3 - p1)

        blk.setExtents(*dimensions)

        s_i = blk.interior

        dx = np.linalg.norm(p2 - p1) / (blk.ni - 1)
        dr = thickness / (blk.nj - 1)
        dtheta = sweep / (blk.nk - 1)

        for j in range(blk.nj):
            for i in range(blk.ni):
                p_ij = np.append(p3 + dx * i * n12 + dr * j * n13, 1)

                blk.array["x"][s_i][i, j, 0] = p_ij[0]
                blk.array["y"][s_i][i, j, 0] = p_ij[1]
                blk.array["z"][s_i][i, j, 0] = p_ij[2]

        xflat = np.reshape(blk.array["x"][s_i][:, :, 0], (blk.ni * blk.nj, 1))
        yflat = np.reshape(blk.array["y"][s_i][:, :, 0], (blk.ni * blk.nj, 1))
        zflat = np.reshape(blk.array["z"][s_i][:, :, 0], (blk.ni * blk.nj, 1))

        pts = np.hstack((xflat, yflat, zflat))
        p = pts - p1
        shape = blk.array["x"][s_i][:, :, 0].shape
        for k in range(1, blk.nk):
            # See http://paulbourke.net/geometry/rotate/
            theta = k * dtheta * np.pi / 180.0
            ct = np.cos(theta)
            st = np.sin(theta)

            q = np.zeros(pts.shape)
            q[:, 0] += (ct + (1 - ct) * n12[0] * n12[0]) * p[:, 0]
            q[:, 0] += ((1 - ct) * n12[0] * n12[1] - n12[2] * st) * p[:, 1]
            q[:, 0] += ((1 - ct) * n12[0] * n12[2] + n12[1] * st) * p[:, 2]

            q[:, 1] += ((1 - ct) * n12[0] * n12[1] + n12[2] * st) * p[:, 0]
            q[:, 1] += (ct + (1 - ct) * n12[1] * n12[1]) * p[:, 1]
            q[:, 1] += ((1 - ct) * n12[1] * n12[2] - n12[0] * st) * p[:, 2]

            q[:, 2] += ((1 - ct) * n12[0] * n12[2] - n12[1] * st) * p[:, 0]
            q[:, 2] += ((1 - ct) * n12[1] * n12[2] + n12[0] * st) * p[:, 1]
            q[:, 2] += (ct + (1 - ct) * n12[2] * n12[2]) * p[:, 2]

            q[:, 0] += p1[0]
            q[:, 1] += p1[1]
            q[:, 2] += p1[2]

            blk.array["x"][s_i][:, :, k] = np.reshape(q[:, 0], shape)
            blk.array["y"][s_i][:, :, k] = np.reshape(q[:, 1], shape)
            blk.array["z"][s_i][:, :, k] = np.reshape(q[:, 2], shape)
