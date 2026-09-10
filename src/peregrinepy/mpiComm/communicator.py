"""
Trading halos between blocks.

A block's halo is filled from whichever block lies across each of its faces,
and that block may be on another rank or on this one. Which faces trade, who
they trade with, and the planes of the block they trade through are all fixed
once the blocks know their neighbors, so they are worked out once here and an
exchange only moves data.
"""

from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request

from ..compute.utils import extractSendBuffer, placeRecvBuffer
from .mpiUtils import getCommRankSize


class Communicator:
    """The halo exchange for one multiBlock.

    A face whose neighbor sits on our own rank is handed what we packed for it
    directly; only a face that leaves the rank costs a message. Most of a well
    packed partition is the former.
    """

    def __init__(self, mb):
        self.comm, self.rank, self.size = getCommRankSize()

        # every face that trades, with the block planes it trades through
        self.trades = []
        # a neighbor on our own rank, as (our face, the face it meets)
        self.local = []
        # a neighbor we have to send to
        self.remote = []

        for blk in mb:
            for face in blk.faces:
                if face.neighbor is None:
                    continue
                self.trades.append((blk, face, self._planeIndices(face)))
                if face.commRank != self.rank:
                    self.remote.append(face)
                    continue
                neighbor = mb.getBlock(face.neighbor)
                assert neighbor is not None, (
                    f"block {blk.nblki} face {face.nface} names rank {self.rank}"
                    f" for neighbor {face.neighbor}, which is not on it"
                )
                self.local.append((face, neighbor.getFace(face.neighborNface)))

    @staticmethod
    def _planeIndices(face):
        """Which plane of the block each buffer layer is, for everything this
        face trades. The compute side indexes with them directly, so they are
        pulled out of the slice objects once rather than on every exchange."""

        def indices(slices):
            return [s for plane in slices for s in plane if isinstance(s, int)]

        return {
            var: (indices(face.sendSlices(var)), indices(face.recvSlices(var)))
            for var in face.commVars
        }

    def exchange(self, varis):
        """Fill every block's halo from its neighbors."""
        if not isinstance(varis, list):
            varis = [varis]
        for var in varis:
            self._exchangeOne(var)

    def _exchangeOne(self, var):
        # what arrives needs somewhere to land before anything is sent
        reqs = [
            self.comm.Irecv(
                [face.array["recvBuffer_" + var], MPIDOUBLE],
                source=face.commRank,
                tag=face.tagR,
            )
            for face in self.remote
        ]

        for blk, face, planes in self.trades:
            self._pack(blk, face, var, planes[var][0])

        # a neighbor on our own rank reads what we packed as it stands
        for face, partner in self.local:
            partner.array["recvBuffer_" + var][:] = face.array["sendBuffer_" + var]

        for face in self.remote:
            buffer = face.array["sendBuffer_" + var]
            self.comm.Send(
                [buffer, buffer.size, MPIDOUBLE], dest=face.commRank, tag=face.tagS
            )

        Request.Waitall(reqs)

        for blk, face, planes in self.trades:
            self._place(blk, face, var, planes[var][1])

        # a face trades under the same tag whatever the variable, so one
        # variable has to land everywhere before the next one goes out
        self.comm.Barrier()

    @staticmethod
    def _pack(blk, face, var, planes):
        """Take this face's planes out of the block and lay them out the way
        its neighbor reads them."""
        # the unoriented planes need somewhere their own shape to sit, and the
        # send buffer is in the neighbor's frame, so the temp buffer holds them
        temp = "tempRecvBuffer_" + var
        extractSendBuffer(
            getattr(blk.cpp, var), getattr(face.cpp, temp), face.cpp, planes
        )
        face.updateHostView(temp)

        send = face.array["sendBuffer_" + var]
        for n in range(len(planes)):
            send[n] = face.orient(face.array[temp][n])

    @staticmethod
    def _place(blk, face, var, planes):
        """Put what arrived into the block's halo."""
        name = "recvBuffer_" + var
        face.updateDeviceView(name)
        placeRecvBuffer(
            getattr(blk.cpp, var), getattr(face.cpp, name), face.cpp, planes
        )
