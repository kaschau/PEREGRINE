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

    def shapeOf(self, name):
        kind, components = self.declared[name]
        return self.shapes[kind] + components

    def setExtents(self, ni, nj, nk):
        """A grid block holds arrays shaped by its extents, so learning them
        is what gives it those arrays, zeroed. One that already has its
        shape keeps what is in it, which is what lets a block be re-sized
        around arrays that have already been rearranged."""
        super().setExtents(ni, nj, nk)
        for name, (kind, components) in self.declared.items():
            shape = self.shapes[kind] + components
            current = getattr(self, name)
            if current is not None and current.shape == shape:
                continue
            array = self.backend.allocate(
                shape, name=name, kind=kind, components=components
            )
            setattr(self, name, array)

    def replace(self, name, values):
        """A new array of these values' shape holding them, in place of the
        old one; what a block re-sized around rearranged contents does."""
        kind, components = self.declared[name]
        array = self.backend.allocate(
            values.shape, name=name, kind=kind, components=components
        )
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
