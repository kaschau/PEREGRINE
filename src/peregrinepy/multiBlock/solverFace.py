import numpy as np

from ..ranges import CellCenterRange
from .gridFace import gridFace


class solverFace(gridFace):
    """A block face of a solver block. It owns only what only a block face
    does: its boundary condition and the values it holds on the face, and
    the buffers a trade with another rank travels in, which the halo
    exchange makes. How the block across it is joined is the topology's;
    everything else it reaches on its block."""

    def __init__(self, nface, blk):
        # its arrays are made where the block's are, as deep a halo
        super().__init__(nface, blk.backend, blk.ng)
        # the block this face bounds
        self.blk = blk
        # the values its boundary condition holds on the face, once given
        self.qBcVals = self.QBcVals = None
        # the buffers its trade travels in, one pair per exchanged array,
        # which the halo exchange makes once the neighbors are known
        for name in blk.mb.exchangedArrays:
            setattr(self, f"sendBuffer_{name}", None)
            setattr(self, f"recvBuffer_{name}", None)

    ###########################################################################
    # The planes of a block array either side of this face, as the kernels
    # walk them: the layer first, layer 0 nearest the face
    ###########################################################################
    def halo(self, array):
        """Gives the halo planes, outward from the face."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[ng - 1 :: -1] if self.amILow else planes[-ng:]

    def interior(self, array, skip=0):
        """Gives the interior planes, inward from the face; :skip: leaves
        out the first, for a node array whose face plane both sides hold."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        if self.amILow:
            return planes[ng + skip : 2 * ng + skip]
        return planes[-(ng + 1 + skip) : -(2 * ng + 1 + skip) : -1]

    def blockFacePlane(self, array):
        """Gives the plane of a cell-face array lying on this block face,
        over the block face proper."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[ng if self.amILow else -(ng + 1)][ng:-ng, ng:-ng]

    ###########################################################################
    # The block face proper: what a plane over it is shaped
    ###########################################################################
    @property
    def bcValuesShape(self):
        """Gives the shape of the values a boundary condition holds on this
        block face proper: one plane, ne wide."""
        _, a, b = CellCenterRange(self.blk.extents, self.blk.ng).haloExtents(
            self.nface, 1
        )
        return (a, b, self.blk.ne)

    def tableValue(self, name):
        """Gives what this block face puts in a table under a name: its own
        -- which side of its block it is, how its neighbor's plane lies,
        its values, its buffers, its rotation, the block's area vectors on
        its axis as S -- else whatever its block puts; under a name ending
        in @neighborFace, what the face it meets on this rank puts under
        the name, and on a face that meets none nothing: no array, or zero
        of a value."""
        name, _, across = name.partition("@")
        if across:
            theirs = self.blk.mb.neighborFace(self)
            if theirs:
                return theirs.tableValue(name)
            return (
                0
                if isinstance(self.tableValue(name), (bool, int, np.integer))
                else None
            )
        own = getattr(self, name, None)
        return own if own is not None else self.blk.tableValue(name)

    # how the neighbor's face plane lies against ours, as the pack kernel
    # takes it
    @property
    def transpose(self):
        return int(self.transposed)

    @property
    def flip0(self):
        return int(0 in self.flipped)

    @property
    def flip1(self):
        return int(1 in self.flipped)

    @property
    def S(self):
        """Gives the block's cell-face area vectors of this face's axis."""
        return getattr(self.blk, f"{self.direction}S")
