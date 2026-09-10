import numpy as np

from .. import bcs
from ..compute import face_
from ..compute.pgkokkos import deep_copy
from ..misc import createViewMirrorArray
from .gridFace import gridFace
from .topologyFace import topologyFace


class solverFace(gridFace):
    faceType = "solver"

    # what a face trades with its neighbor, and the shape each trade takes
    commVars = {
        "x": "node",
        "y": "node",
        "z": "node",
        "q": "state",
        "Q": "state",
        "dqdx": "gradient",
        "dqdy": "gradient",
        "dqdz": "gradient",
        "phi": "switch",
    }

    def __init__(self, nface, ng):
        # the compute object must exist before anything forwards to it
        self.cpp = face_()
        super().__init__(nface)

        self._ng = ng
        # the block this face bounds, once it knows how big it is
        self.blockExtents = None
        self.ne = None

        # Face slices: the halo planes this face owns, outermost first, the
        # first plane inside the block, and the planes a halo reflects about it
        if self.amILow:
            s0 = range(ng - 1, -1, -1)
            s1 = ng
            s2 = range(ng + 1, 2 * ng + 1)
        else:
            s0 = range(-ng, 0)
            s1 = -(ng + 1)
            s2 = range(-(ng + 2), -(2 * ng + 2), -1)
        self.s0_ = [self._facePlane(i) for i in s0]
        self.s1_ = self._facePlane(s1)
        self.s2_ = [self._facePlane(i) for i in s2]

        # arrays that faces save
        self.declare("qBcVals", "QBcVals", kind="bcValues")
        for var, kind in self.commVars.items():
            self.declare(f"sendBuffer_{var}", kind=f"{kind}Send")
            self.declare(f"recvBuffer_{var}", f"tempRecvBuffer_{var}", kind=kind)

        # Boundary function
        self.bcFunc = bcs.getBc("adiabaticSlipWall").kernel()

        # MPI variables
        self.commRank = None
        self.tagS = None
        self.tagR = None

        # how our neighbor's face plane lies against ours, and the planes of
        # the block that travel between us
        self._transposed = None
        self._flipped = None
        self._sendSlices = None
        self._recvSlices = None

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
            "node": (ng,) + node,
            "nodeSend": (ng,) + theirNode,
            "state": (ng,) + cell + (ne,),
            "stateSend": (ng,) + theirCell + (ne,),
            # a gradient is only ever wanted one cell past the block
            "gradient": (1,) + cell + (ne,),
            "gradientSend": (1,) + theirCell + (ne,),
            "switch": (1,) + cell + (3,),
            "switchSend": (1,) + theirCell + (3,),
            # what a bc holds across the face itself
            "bcValues": cell + (ne,),
        }

    def allocate(self, *names):
        """A solver face's arrays are Kokkos views, with a host mirror the
        numpy array wraps."""
        for name in names:
            createViewMirrorArray(self, name, list(self.shapeOf(name)))

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
            f"{role}Buffer_{var}"
            for var in self.commVars
            for role in ("send", "recv", "tempRecv")
        ]

    ###########################################################################
    # Talking to our neighbor
    ###########################################################################
    def setCommunication(self, nblki):
        """Everything this face needs to trade halos with its neighbor: how
        the neighbor's plane lies against ours, which planes of the block go
        out and where the ones that arrive are put, and the buffers they
        travel in."""
        assert (
            self.blockExtents is not None
        ), "Must get grid before setting block communications."

        self._setOrient()
        self._setSlices()
        self.allocate(*self.commArrays)

        # Unique tags
        self.tagR = int(nblki * 6 + self.nface)
        self.tagS = int(self.neighbor * 6 + self.neighborNface)

    def orient(self, plane):
        """A plane of ours, laid out the way our neighbor reads it."""
        if self._transposed:
            plane = np.moveaxis(plane, (0, 1), (1, 0))
        return np.flip(plane, self._flipped) if self._flipped else plane

    def sendSlices(self, var):
        """The planes of the block that go out in this face's send buffer."""
        return self._sendSlices[self.commVars[var]]

    def recvSlices(self, var):
        """The planes of the block that what arrives is placed into."""
        return self._recvSlices[self.commVars[var]]

    def _facePlane(self, index):
        """The index-plane of a block array normal to this face."""
        return (slice(None),) * self.myAxis + (index,)

    def _setOrient(self):
        """How a plane of ours is laid out in our neighbor's frame: which of
        our two face axes it reads first, and which way round it reads each."""
        # our neighbor's orientation names one of our axes for each of its
        # own; the one naming the normal of the face we share is not in the
        # plane, so drop it
        theirNormal = (self.neighborNface - 1) // 2
        theirPlane = [
            self.signedAxis(code)
            for m, code in enumerate(self.neighborOrientation)
            if m != theirNormal
        ]
        ourPlane = [m for m in range(3) if m != self.myAxis]

        self._transposed = theirPlane[0][0] == ourPlane[1]
        self._flipped = tuple(
            m for m, (_, counterAligned) in enumerate(theirPlane) if counterAligned
        )

    def _setSlices(self):
        """Which planes of the block go out, and where the ones that arrive
        are put.

        The recv list always runs from the smallest index to the largest.
        The send list starts out the same way, and is reversed when our
        neighbor's axis runs against ours, so that what arrives is already
        in the order it goes in.

              index -------------------------->
         o----------o----------o|x----------x----------x
         |          |           |           |          |
         | recv[0]  |  recv[1]  |  send[0]  |  send[1] |
         |          |           |           |          |
         o----------o----------o|x----------x----------x
        """
        ng = self.ng
        # these index the compute side views as well as ours, so they count
        # from the near end of the axis; Kokkos has no index from the far end
        nNodes = self.blockExtents[self.myAxis] + 2 * ng
        nCells = nNodes - 1
        if self.amILow:
            nodeOut, nodeIn = range(ng + 1, 2 * ng + 1), range(0, ng)
            cellOut, cellIn = range(ng, 2 * ng), range(0, ng)
        else:
            nodeOut = range(nNodes - (2 * ng + 1), nNodes - (ng + 1))
            nodeIn = range(nNodes - ng, nNodes)
            cellOut = range(nCells - 2 * ng, nCells - ng)
            cellIn = range(nCells - ng, nCells)

        nodeSend = [self._facePlane(i) for i in nodeOut]
        nodeRecv = [self._facePlane(i) for i in nodeIn]
        cellSend = [self._facePlane(i) for i in cellOut]
        cellRecv = [self._facePlane(i) for i in cellIn]

        # which plane is the one nearest the block must be picked before the
        # send order is reversed
        if self.amILow:
            firstSend, firstRecv = [cellSend[0]], [cellRecv[-1]]
        else:
            firstSend, firstRecv = [cellSend[-1]], [cellRecv[0]]

        _, counterAligned = self.signedAxis(self.orientation[self.myAxis])
        if counterAligned:
            nodeSend.reverse()
            cellSend.reverse()

        self._sendSlices = {
            "node": nodeSend,
            "state": cellSend,
            "gradient": firstSend,
            "switch": firstSend,
        }
        self._recvSlices = {
            "node": nodeRecv,
            "state": cellRecv,
            "gradient": firstRecv,
            "switch": firstRecv,
        }

    ###########################################################################
    # What a solver face keeps on the device
    ###########################################################################
    @topologyFace.bcType.setter
    def bcType(self, value):
        topologyFace.bcType.fset(self, value)
        self.bcFunc = bcs.getBc(self.bcType).kernel()

    @gridFace.periodicAxis.setter
    def periodicAxis(self, axis):
        gridFace.periodicAxis.fset(self, axis)
        if axis is not None and self.bcType.startswith("periodicRot"):
            self.updateDeviceView(["periodicRotMatrixUp", "periodicRotMatrixDown"])

    def updateDeviceView(self, vars):
        if isinstance(vars, str):
            vars = [vars]
        for var in vars:
            deep_copy(getattr(self.cpp, var), self.mirror[var])

    def updateHostView(self, vars):
        if isinstance(vars, str):
            vars = [vars]
        for var in vars:
            deep_copy(self.mirror[var], getattr(self.cpp, var))

    @property
    def ng(self):
        return self.cpp._ng

    @ng.setter
    def ng(self, value):
        self.cpp._ng = value

    @property
    def _ng(self):
        return self.cpp._ng

    @_ng.setter
    def _ng(self, value):
        self.cpp._ng = value

    @property
    def _nface(self):
        return self.cpp._nface

    @_nface.setter
    def _nface(self, value):
        self.cpp._nface = value
