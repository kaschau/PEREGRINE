import numpy as np

from .. import bcs
from ..ranges import CellCenterRange
from .gridFace import gridFace
from .topologyFace import topologyFace


class solverFace(gridFace):
    """A block face of a solver block. It owns only what only a block face
    does: its boundary condition and the values it holds on the face, and
    its trade with the block across it -- how that block's plane lies
    against this one and the buffers the trade travels in, which the halo
    exchange makes. Everything else it reaches on its block."""

    def __init__(self, nface, ng, blk):
        # its arrays are made where the block's are
        super().__init__(nface, blk.backend)
        self.ng = ng
        # the block this face bounds
        self.blk = blk
        # its boundary condition, an object of the type it carries, and the
        # values that condition holds on the face, once given
        self.bc = bcs.getBc(self.bcType)(self)
        self.qBcVals = self.QBcVals = None
        # the buffers its trade travels in, one pair per exchanged array,
        # which the halo exchange makes once the neighbors are known
        for name, _ in blk.mb.exchanged():
            setattr(self, f"sendBuffer_{name}", None)
            setattr(self, f"recvBuffer_{name}", None)
        # how our neighbor's face plane lies against ours: whether its two
        # axes cross ours, and which of them run backwards
        self._transposed = None
        self._flipped = None

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
        its axis as S -- else whatever its block puts."""
        own = getattr(self, name, None)
        return own if own is not None else self.blk.tableValue(name)

    @property
    def transpose(self):
        """Says whether the neighbor's face axes cross ours, as the pack
        kernel takes it."""
        return int(bool(self._transposed))

    @property
    def flip0(self):
        return int(0 in (self._flipped or ()))

    @property
    def flip1(self):
        return int(1 in (self._flipped or ()))

    @property
    def S(self):
        """Gives the block's cell-face area vectors of this face's axis."""
        return getattr(self.blk, f"{self.direction}S")

    ###########################################################################
    # Talking to our neighbor
    ###########################################################################
    def setCommunication(self):
        """Settles how the neighbor's plane lies against ours: which of our
        two face axes it reads first, and which way round it reads each."""
        self._transposed, self._flipped = self.neighborPlaneAlignment

    @topologyFace.bcType.setter
    def bcType(self, value):
        topologyFace.bcType.fset(self, value)
        self.bc = bcs.getBc(value)(self)
