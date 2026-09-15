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

from ..abi import lib
from ..kernel import BoundKernel
from ..table import Table
from .mpiUtils import getCommRankSize


class Communicator:
    """The halo exchange for one multiBlock.

    A face whose neighbor sits on our own rank is handed what we packed for it
    directly; only a face that leaves the rank costs a message. Most of a well
    packed partition is the former.
    """

    def __init__(self, table, thtrdat):
        # the pack and unpack, every trading face in one call
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
        # per variable: the pack and unpack tables over every trade, and
        # where a remote face's message lands, all kept once made
        self.tables = {}
        self.landings = {}

    def connect(self, faces):
        """Which of the (block, face) pairs trade and with whom, once the
        blocks know their neighbors."""
        faces = list(faces)
        blocks = {blk.nblki: blk for blk, _ in faces}
        self.trades, self.local, self.remote = [], [], []
        self.tables, self.landings = {}, {}
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

    def _tables(self, var):
        """The pack and unpack tables for one variable: each trade's block
        array, its buffers and which planes it trades. A neighbor on our own
        rank is unpacked straight out of what it packed; a message lands in
        the face's own buffer first."""
        if var not in self.tables:
            partners = dict(self.local)
            pack, unpack = Table(), Table()
            for index, (blk, face) in enumerate(self.trades):
                nLayer, skip = face.tradeLayers(var)
                pack.register(index, "view", getattr(blk, var))
                pack.register(index, "buffer", getattr(face, "sendBuffer_" + var))
                pack.setInt(index, "nface", face.nface)
                pack.setInt(index, "nLayer", nLayer)
                pack.setInt(index, "skip", skip)
                pack.setInt(index, "transpose", int(face._transposed))
                pack.setInt(index, "flip0", int(0 in face._flipped))
                pack.setInt(index, "flip1", int(1 in face._flipped))
                if face in partners:
                    source = getattr(partners[face], "sendBuffer_" + var)
                else:
                    source = getattr(face, "recvBuffer_" + var)
                unpack.register(index, "view", getattr(blk, var))
                unpack.register(index, "buffer", source)
                unpack.setInt(index, "nface", face.nface)
                unpack.setInt(index, "nLayer", nLayer)
            self.tables[var] = (pack, unpack)
        return self.tables[var]

    def _ndim(self, var):
        """How many dimensions a variable's arrays have, which its trades are
        all of; not an MPI rank."""
        return len(getattr(self.trades[0][0], var).shape)

    def _landing(self, face, var):
        key = (face, var)
        if key not in self.landings:
            # laid out as the device buffer is, so the bytes land where they go
            buffer = getattr(face, "recvBuffer_" + var)
            self.landings[key] = np.empty(buffer.shape, buffer.dtype, buffer.order)
        return self.landings[key]

    def _exchangeOne(self, var):
        send, recv = "sendBuffer_" + var, "recvBuffer_" + var
        pack, unpack = self._tables(var)

        # what arrives needs somewhere to land before anything is sent
        landing = {face: self._landing(face, var) for face in self.remote}
        reqs = [
            self.comm.Irecv(
                [landing[face], MPIDOUBLE], source=face.commRank, tag=face.tagR
            )
            for face in self.remote
        ]

        # the pack turns the plane onto the neighbor's frame as it goes, so
        # nothing here has to leave the device; a rank with nothing to trade
        # still meets the others at the barrier
        if self.trades:
            self.pack(pack, ndim=self._ndim(var))

        # only a message has to go through the host: every snapshot is asked
        # for, one wait covers them all, and what lands is set back in with
        # the unpack following it in order
        snapshots = {face: getattr(face, send).get(wait=False) for face in self.remote}
        if snapshots:
            lib.pgFence()
        for face, buffer in snapshots.items():
            self.comm.Send(
                [buffer, buffer.size, MPIDOUBLE], dest=face.commRank, tag=face.tagS
            )

        Request.Waitall(reqs)
        for face in self.remote:
            getattr(face, recv).set(landing[face], wait=False)

        if self.trades:
            self.unpack(unpack, ndim=self._ndim(var))

        # a face trades under the same tag whatever the variable, so one
        # variable has to land everywhere before the next one goes out
        self.comm.Barrier()
