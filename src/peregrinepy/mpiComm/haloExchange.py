"""
Trading halos between blocks.

A block's halo is filled from whichever block lies across each of its faces,
and that block may be on another rank or on this one. Which faces trade, who
they trade with, and the planes of the block they trade through are all fixed
once the blocks know their neighbors, so they are worked out once here and an
exchange only moves data.

The halo exchange moves halos and nothing else: it is given its pack and
unpack kernels, builds its trade tables from Trade entries, and is told to
forget when a block or a face re-makes its arrays.
"""

import numpy as np
from mpi4py.MPI import DOUBLE as MPIDOUBLE
from mpi4py.MPI import Request

from ..abi import lib
from ..table import Table
from .mpiUtils import getCommRankSize


class Trade:
    """One face's trade of one variable through one buffer: what a pack or
    an unpack takes for it. A block array's planes go out through the face's
    send buffer, laid out for the neighbor, and come in from whichever
    buffer the neighbor's arrived in."""

    def __init__(self, blk, face, var, buffer):
        self.blk, self.face = blk, face
        self.view = getattr(blk, var)
        self.buffer = buffer
        self.nLayer, self.skip = face.tradeLayers(var)

    def column(self, name):
        if name == "nface":
            return self.face.nface
        if name == "transpose":
            return int(self.face._transposed)
        if name in ("flip0", "flip1"):
            return int(int(name[-1]) in self.face._flipped)
        return getattr(self, name)


class HaloExchange:
    """The halo exchange for one multiBlock.

    A face whose neighbor sits on our own rank is handed what we packed for it
    directly; only a face that leaves the rank costs a message. Most of a well
    packed partition is the former.
    """

    def __init__(self, pack, unpack, tileSize, backend):
        # the pack and unpack, every trading face in one call; the trade
        # tables are kept where the kernels run
        self.pack, self.unpack = pack, unpack
        self.tileSize, self.backend = tileSize, backend
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

    def forget(self):
        """A block or a face re-made its arrays: every trade table and
        landing is read again."""
        self.tables, self.landings = {}, {}

    def _tables(self, var):
        """The pack and unpack tables for one variable, a trade per face. A
        neighbor on our own rank is unpacked straight out of what it packed;
        a message lands in the face's own buffer first."""
        if var not in self.tables:
            partners = dict(self.local)
            pack, unpack = [], []
            for blk, face in self.trades:
                pack.append(Trade(blk, face, var, getattr(face, "sendBuffer_" + var)))
                if face in partners:
                    arrival = getattr(partners[face], "sendBuffer_" + var)
                else:
                    arrival = getattr(face, "recvBuffer_" + var)
                unpack.append(Trade(blk, face, var, arrival))
            self.tables[var] = (
                Table(pack, self.tileSize, self.backend),
                Table(unpack, self.tileSize, self.backend),
            )
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
