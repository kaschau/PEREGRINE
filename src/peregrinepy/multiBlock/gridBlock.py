import numpy as np
from .gridFace import gridFace
from .metricsMixin import MetricsMixin
from .topologyBlock import topologyBlock


class gridBlock(topologyBlock, MetricsMixin):
    """
    gridBlock object holds all the information that a grid
    would need to know about a block.
    """

    blockType = "grid"

    def __init__(self, nblki, ng=0):
        self.ng = ng

        super().__init__(nblki)

        #########################################################
        # Data arrays
        #########################################################
        # Python side data
        self.array = {}
        # Kokkos mirrors (only used for solverBlocks)
        self.mirror = {}
        # what each array's shape will be, once the extents are known
        self.declared = {}

        self.declare("x", "y", "z", kind="node")
        # cell centers are as much as a block with no solution on it can work
        # out; the rest of the metrics are a solverBlock's
        self.declare("xc", "yc", "zc", kind="cell")

    def splitAlong(self, axis, cutIndex):
        """A grid block holds the coordinates its cut splits in two. The two
        halves share the plane they are cut on."""
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis] = slice(0, cutIndex + 1)
        high[axis] = slice(cutIndex, None)
        return {
            var: (
                np.copy(self.array[var][tuple(low)]),
                np.copy(self.array[var][tuple(high)]),
            )
            for var in ("x", "y", "z")
        }

    def _newFace(self, nface):
        return gridFace(nface)

    ###########################################################################
    # The arrays a block has, and how big they are
    ###########################################################################
    def declare(self, *names, kind, components=None):
        """Say an array exists and what shape it will take, before this block
        knows its extents. Nothing else may be put in array."""
        for name in names:
            self.declared[name] = (kind, components)
            self.array[name] = None
            self.mirror[name] = None

    @property
    def shapes(self):
        """What each kind of array is shaped, for this block's extents."""
        ng, ni, nj, nk = self.ng, self.ni, self.nj, self.nk
        return {
            "node": (ni + 2 * ng, nj + 2 * ng, nk + 2 * ng),
            "cell": (ni + 2 * ng - 1, nj + 2 * ng - 1, nk + 2 * ng - 1),
            "iface": (ni + 2 * ng, nj + 2 * ng - 1, nk + 2 * ng - 1),
            "jface": (ni + 2 * ng - 1, nj + 2 * ng, nk + 2 * ng - 1),
            "kface": (ni + 2 * ng - 1, nj + 2 * ng - 1, nk + 2 * ng),
        }

    def shapeOf(self, name):
        kind, components = self.declared[name]
        shape = self.shapes[kind]
        return shape + (components,) if components else shape

    def setExtents(self, ni, nj, nk):
        """A grid block holds arrays shaped by its extents, so learning them
        is what gives it those arrays."""
        super().setExtents(ni, nj, nk)
        self.allocate()

    def allocate(self):
        for name in self.declared:
            self.array[name] = np.zeros(self.shapeOf(name))

    @property
    def interior(self):
        """The slice of this block's arrays that is not halo."""
        return np.s_[:, :, :]

    @property
    def baseNodeSlab(self):
        """Where this block's nodes sit in the base block it is a piece of,
        which is how the grid and result files are indexed. baseSlice counts
        nodes inclusively."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]

    @property
    def baseCellSlab(self):
        """The same, for this block's cells. A block of ni nodes spans ni-1
        cells, so nodes i0..i1 are cells i0..i1-1."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0:k1, j0:j1, i0:i1]

    def updateDeviceView(self, vars):
        """No device to push to."""

    def updateHostView(self, vars):
        """No device to pull from."""
