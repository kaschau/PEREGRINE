import numpy as np

from ..abi import HostStorageMixin
from .gridFace import gridFace
from .metricsMixin import MetricsMixin
from .topologyBlock import topologyBlock


class gridBlock(topologyBlock, MetricsMixin, HostStorageMixin):
    """
    gridBlock object holds all the information that a grid
    would need to know about a block.
    """

    def __init__(self, nblki, ng=0):
        self.ng = ng

        super().__init__(nblki)

        # every array is an attribute named for it, shaped once the extents are
        # known; what a block declares is all it may hold
        self.declared = {}

        self.declare("nodes", kind="node", components=3)
        # cell centers are as much as a block with no solution on it can work
        # out; the rest of the metrics are a solverBlock's
        self.declare("cells", kind="cell", components=3)

    def splitAlong(self, axis, cutIndex):
        """A grid block holds the coordinates its cut splits in two. The two
        halves share the plane they are cut on."""
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis] = slice(0, cutIndex + 1)
        high[axis] = slice(cutIndex, None)
        return {
            "nodes": (
                np.copy(self.nodes[tuple(low)]),
                np.copy(self.nodes[tuple(high)]),
            )
        }

    def _newFace(self, nface):
        return gridFace(nface)

    ###########################################################################
    # The arrays a block has, and how big they are
    ###########################################################################
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

    def setExtents(self, ni, nj, nk):
        """A grid block holds arrays shaped by its extents, so learning them
        is what gives it those arrays."""
        super().setExtents(ni, nj, nk)
        self.allocate()

    def allocate(self):
        """Give every declared array the memory its shape asks for. One that
        already has that shape keeps what is in it, which is what lets a block
        be re-sized around arrays that have already been rearranged."""
        for name in self.declared:
            shape = self.shapeOf(name)
            current = getattr(self, name)
            if current is not None and current.shape == shape:
                continue
            setattr(self, name, self._new(shape))
            self._placed(name)

    def _placed(self, name):
        """What a kind of block does with an array it just made; nothing here."""

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
