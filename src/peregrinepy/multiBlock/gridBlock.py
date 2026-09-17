import numpy as np

from .gridFace import gridFace
from .metricsMixin import MetricsMixin
from .topologyBlock import topologyBlock


class gridBlock(topologyBlock, MetricsMixin):
    """A block with coordinates and the arrays that follow. Each array the
    multiBlock declared is stored as blk.<name>, made on the multiBlock's
    backend at setExtents; a block declares nothing of its own."""

    def __init__(self, nblki, mb):
        self.ng = mb.ng
        # what the multiBlock says a block holds, and the backend its arrays
        # are made on
        self.declared = mb.arrays
        self.backend = mb.backend
        super().__init__(nblki)
        # every array is an attribute named for it, made once the extents are known
        for name in self.declared:
            setattr(self, name, None)

    def splitAlong(self, axis, cutIndex):
        """A grid block holds the coordinates its cut splits in two. The two
        halves share the plane they are cut on."""
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis] = slice(0, cutIndex + 1)
        high[axis] = slice(cutIndex, None)
        nodes = self.nodes.get()
        return {"nodes": (nodes[tuple(low)], nodes[tuple(high)])}

    def _newFace(self, nface):
        return gridFace(nface, self.backend)

    ###########################################################################
    # The arrays a block has
    ###########################################################################
    @property
    def extents(self):
        """Gives this block's extents, its nodes per axis: (ni, nj, nk)."""
        return (self.ni, self.nj, self.nk)

    def setExtents(self, ni, nj, nk):
        """Sizes this block, which is what gives it its arrays, zeroed: each
        declared one made for this block by its kind."""
        super().setExtents(ni, nj, nk)
        for name in self.declared:
            setattr(self, name, self._make(name))

    def _make(self, name):
        kind, components, rangeArgs = self.declared[name]
        return kind(self, name, components, **rangeArgs)

    def replace(self, name, values):
        """Puts a new array holding these values in place of the old one of
        this name: a new arrayInfo for the tables that read it."""
        array = self._make(name)
        array.set(values)
        setattr(self, name, array)

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
