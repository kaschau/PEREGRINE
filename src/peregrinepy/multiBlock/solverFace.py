import numpy as np

from .. import bcs
from .gridFace import gridFace
from .topologyFace import topologyFace


class solverFace(gridFace):
    # what a face trades with its neighbor, and the shape each trade takes
    commVars = {
        "nodes": "node",
        "Q": "state",
        "grads": "gradient",
    }

    def __init__(self, nface, ng, blk):
        # its arrays are made where the block's are
        super().__init__(nface, blk.backend)

        self.ng = ng
        # the block this face bounds
        self.blk = blk
        # its boundary condition, an object of the type it carries
        self.bc = bcs.getBc(self.bcType)(self)
        # the block this face bounds, once it knows how big it is
        self.blockExtents = None
        self.ne = None

        # arrays that faces save
        self.declare("qBcVals", "QBcVals", kind="bcValues")
        for var, kind in self.commVars.items():
            self.declare(f"sendBuffer_{var}", kind=f"{kind}Send")
            self.declare(f"recvBuffer_{var}", kind=kind)

        # MPI variables
        self.tagS = None
        self.tagR = None

        # how our neighbor's face plane lies against ours, and the planes of
        # the block that travel between us
        self._transposed = None
        self._flipped = None

    ###########################################################################
    # The planes of a block array either side of this face, as the kernels
    # walk them: the layer first, layer 0 nearest the face
    ###########################################################################
    def halo(self, array):
        """The halo planes, outward from the face."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[ng - 1 :: -1] if self.amILow else planes[-ng:]

    def interior(self, array, skip=0):
        """The interior planes, inward from the face; :skip: leaves out the
        first, for a node array whose face plane both sides hold."""
        ng = self.ng
        planes = np.moveaxis(array, self.myAxis, 0)
        if self.amILow:
            return planes[ng + skip : 2 * ng + skip]
        return planes[-(ng + 1 + skip) : -(2 * ng + 1 + skip) : -1]

    def boundary(self, array):
        """The one plane of a face array on the face itself."""
        planes = np.moveaxis(array, self.myAxis, 0)
        return planes[self.ng if self.amILow else -(self.ng + 1)]

    ###########################################################################
    # The arrays a face has, and how big they are
    ###########################################################################
    @property
    def shapes(self):
        """What each kind of array is shaped, for the block this face bounds.
        A buffer is laid out as the halo layer it carries, then the two axes
        of the face, then whatever sits at each point of it."""
        shapes = super().shapes
        # a face is built long before its block is sized, and its rotation
        # matrices do not wait for that
        if self.blockExtents is None:
            return shapes

        ng, ne = self.ng, self.ne
        node = tuple(
            n + 2 * ng for m, n in enumerate(self.blockExtents) if m != self.myAxis
        )
        cell = tuple(n - 1 for n in node)
        # a send buffer is packed the way its neighbor reads it, and a
        # neighbor whose face axes cross ours reads the plane the other way
        theirNode = node[::-1] if self._transposed else node
        theirCell = cell[::-1] if self._transposed else cell

        return shapes | {
            "node": (ng,) + node + (3,),
            "nodeSend": (ng,) + theirNode + (3,),
            "state": (ng,) + cell + (ne,),
            "stateSend": (ng,) + theirCell + (ne,),
            # a gradient is only ever wanted one cell past the block; there is
            # none of the pressure
            "gradient": (1,) + cell + (ne - 1, 3),
            "gradientSend": (1,) + theirCell + (ne - 1, 3),
            # what a bc holds across the face itself
            "bcValues": cell + (ne,),
        }

    def allocate(self, *names):
        super().allocate(*names)
        # a new array is a new record for the bcs that run over this face
        self.blk.mb.facesChanged = True

    def column(self, name):
        """What a face table takes from this face: which side of its block
        it is, its own arrays, the block's area vectors on its axis as S and
        its rotation as rot, then the block's arrays."""
        if name == "nface":
            return self.nface
        if name == "S":
            return getattr(self.blk, f"{self.direction}S")
        if name == "rot":
            return self.periodicRotMatrix
        own = getattr(self, name) if name in self.declared else None
        if own is not None:
            return own
        return self.blk.column(name)

    def setExtents(self, ni, nj, nk, ne):
        """The block this face bounds is this big, so this face's arrays can
        be shaped. Nothing is allocated yet -- which of them this face needs
        depends on what it turns out to be."""
        self.blockExtents = (int(ni), int(nj), int(nk))
        self.ne = int(ne)

    @property
    def commArrays(self):
        """Every buffer this face trades through."""
        return [
            f"{role}Buffer_{var}" for var in self.commVars for role in ("send", "recv")
        ]

    ###########################################################################
    # Talking to our neighbor
    ###########################################################################
    def setCommunication(self, nblki):
        """Everything this face needs to trade halos with its neighbor: how
        the neighbor's plane lies against ours, and the buffers the trade
        travels in."""
        assert (
            self.blockExtents is not None
        ), "Must get grid before setting block communications."

        # how a plane of ours is laid out in our neighbor's frame: which of
        # our two face axes it reads first, and which way round it reads each
        self._transposed, self._flipped = self.neighborPlaneAlignment
        self.allocate(*self.commArrays)

        # Unique tags
        self.tagR = int(nblki * 6 + self.nface)
        self.tagS = int(self.neighbor * 6 + self.neighborNface)

    def tradeLayers(self, var):
        """How many planes of the block this face trades for an array, and
        how far in the first one is: a node array's face plane is shared by
        both sides, so its trade starts past it."""
        kind = self.commVars[var]
        if kind == "node":
            return self.ng, 1
        if kind == "state":
            return self.ng, 0
        return 1, 0

    ###########################################################################
    # What a solver face keeps on the device
    ###########################################################################
    @topologyFace.bcType.setter
    def bcType(self, value):
        topologyFace.bcType.fset(self, value)
        self.bc = bcs.getBc(value)(self)
        # the face tables follow its bcType
        self.blk.mb.facesChanged = True
