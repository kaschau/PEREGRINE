"""
Trading halos between blocks.

A block's halo is filled from whichever block lies across each of its faces,
and that block may be on another rank or on this one. Which faces trade, who
they trade with, and the planes of the block they trade through are all fixed
once the blocks know their neighbors, so they are worked out once here and an
exchange only moves data.
"""

import numpy as np
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request

from ..kernel import BoundKernel
from .mpiUtils import getCommRankSize


class Communicator:
    """The halo exchange for one multiBlock.

    A face whose neighbor sits on our own rank is handed what we packed for it
    directly; only a face that leaves the rank costs a message. Most of a well
    packed partition is the former.
    """

    def __init__(self, table, thtrdat):
        # the pack and unpack, one face at a time
        self.pack = BoundKernel(table, thtrdat, "utils/extractSendBuffer.cpp")
        self.unpack = BoundKernel(table, thtrdat, "utils/placeRecvBuffer.cpp")
        self.kernels = [self.pack, self.unpack]
        self.comm, self.rank, self.size = getCommRankSize()

        # every face that trades, with the block planes it trades through
        self.trades = []
        # a neighbor on our own rank, as (our face, the face it meets)
        self.local = []
        # a neighbor we have to send to
        self.remote = []

    def connect(self, faces):
        """Which of the (block, face) pairs trade and with whom, once the
        blocks know their neighbors."""
        faces = list(faces)
        blocks = {blk.nblki: blk for blk, _ in faces}
        self.trades, self.local, self.remote = [], [], []
        for blk, face in faces:
            if face.neighbor is None:
                continue
            self.trades.append((blk, face))
            if face.commRank != self.rank:
                self.remote.append(face)
                continue
            neighbor = blocks.get(face.neighbor)
            assert neighbor is not None, (
                f"block {blk.nblki} face {face.nface} names rank {self.rank}"
                f" for neighbor {face.neighbor}, which is not on it"
            )
            self.local.append((face, neighbor.getFace(face.neighborNface)))

    def exchange(self, varis):
        """Fill every block's halo from its neighbors."""
        if not isinstance(varis, list):
            varis = [varis]
        for var in varis:
            self._exchangeOne(var)

    def _exchangeOne(self, var):
        send, recv = "sendBuffer_" + var, "recvBuffer_" + var

        # what arrives needs somewhere to land before anything is sent
        landing = {face: np.empty(getattr(face, recv).shape) for face in self.remote}
        reqs = [
            self.comm.Irecv(
                [landing[face], MPIDOUBLE], source=face.commRank, tag=face.tagR
            )
            for face in self.remote
        ]

        # the pack turns the plane onto the neighbor's frame as it goes, so
        # nothing here has to leave the device
        for blk, face in self.trades:
            nLayer, skip = face.tradeLayers(var)
            self.pack(
                view=getattr(blk, var),
                buffer=getattr(face, send),
                nface=face.nface,
                nLayer=nLayer,
                skip=skip,
                transpose=int(face._transposed),
                flip0=int(0 in face._flipped),
                flip1=int(1 in face._flipped),
            )

        # a neighbor on our own rank reads what we packed as it stands
        for face, partner in self.local:
            getattr(partner, recv).copyFrom(getattr(face, send))

        # only a message has to go through the host: a snapshot out, and what
        # lands set back in
        for face in self.remote:
            buffer = getattr(face, send).get()
            self.comm.Send(
                [buffer, buffer.size, MPIDOUBLE], dest=face.commRank, tag=face.tagS
            )

        Request.Waitall(reqs)
        for face in self.remote:
            getattr(face, recv).set(landing[face])

        for blk, face in self.trades:
            self.unpack(
                view=getattr(blk, var),
                buffer=getattr(face, recv),
                nface=face.nface,
                nLayer=face.tradeLayers(var)[0],
            )

        # a face trades under the same tag whatever the variable, so one
        # variable has to land everywhere before the next one goes out
        self.comm.Barrier()
