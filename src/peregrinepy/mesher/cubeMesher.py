import numpy as np

from .baseMesher import BaseMesher


class CubeMesher(BaseMesher):
    """A rectangular box, split into a lattice of boxes."""

    name = "cube"

    def __init__(
        self,
        origin=[0, 0, 0],
        lengths=[1, 1, 1],
        periodic=[False, False, False],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.origin = list(origin)
        self.lengths = list(lengths)
        self.periodic = list(periodic)

        # where each block of the lattice starts and ends
        self.edges = [
            np.linspace(origin[n], origin[n] + lengths[n], self.mbDims[n] + 1)
            for n in range(3)
        ]

    @property
    def periodicAxes(self):
        return tuple(self.periodic)

    def shapeBlock(self, blk, i, j, k):
        corner = [self.edges[n][m] for n, m in enumerate((i, j, k))]
        lengths = [
            self.edges[n][m + 1] - self.edges[n][m] for n, m in enumerate((i, j, k))
        ]
        self._cube(blk, corner, lengths, self.dimsPerBlock)

    def setPeriodicFaces(self, blk, i, j, k):
        for face in blk.faces:
            if face.bcType != "periodicTrans":
                continue
            # the low face of an axis takes its halo from the high end, so it
            # is moved back down the axis, and the high face the other way
            axis = face.myAxis
            span = -self.lengths[axis] if face.amILow else self.lengths[axis]
            face.setPeriodic(translation=[span if n == axis else 0.0 for n in range(3)])

    def _cube(self, blk, origin, lengths, dimensions):
        """Function to populate the coordinate arrays of a provided peregrinepy.block in the shape of a cube with prescribed location, extents, and discretization.
        If the input multiBlock object is a restart block the shape and size of the flow data arrays are also updated.

        Parameters
        ----------

        blk : peregrinepy.blocks.grid_block (or one of its descendants)

        origin : list, tuple
           List/tuple of length 3 containing the location of the origin of the cube to be created

        lengths : list, tuple
           List/tuple of length 3 containing the extents in x, y, and z of the cube relative to the origin

        dimensions : list, tuple
           List/tuple of length 3 containing discretization (nx,nj,nk) in each dimension of the cube.

        Returns
        -------
        None
            Updates attributes of parameter blk.

        """
        blk.setExtents(*dimensions)

        x = np.linspace(
            origin[0], origin[0] + lengths[0], dimensions[0], dtype=np.float64
        )
        y = np.linspace(
            origin[1], origin[1] + lengths[1], dimensions[1], dtype=np.float64
        )
        z = np.linspace(
            origin[2], origin[2] + lengths[2], dimensions[2], dtype=np.float64
        )

        nodes = blk.nodes.get()
        nodes[blk.interior] = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        blk.nodes.set(nodes)
