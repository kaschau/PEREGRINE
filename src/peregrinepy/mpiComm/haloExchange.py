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

from ..backend.abi import lib
from ..backend.array import Array
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

    def __init__(self, pack, unpack, backend):
        # the pack and unpack, every trading face in one call; the trade
        # tables are kept where the kernels run
        self.pack, self.unpack = pack, unpack
        self.backend = backend
        self.comm, self.rank, self.size = getCommRankSize()

        # every face that trades, with the block planes it trades through
        self.trades = []
        # a neighbor on our own rank, as (our face, the face it meets)
        self.local = []
        # a neighbor we have to send to
        self.remote = []
        # per variable: the pack and unpack tables over every trade, and the
        # pools the remote faces' buffers sit in, all kept once made
        self.tables = {}
        self.pools = {}

    def connect(self, faces):
        """Which of the (block, face) pairs trade and with whom, once the
        blocks know their neighbors."""
        faces = list(faces)
        blocks = {blk.nblki: blk for blk, _ in faces}
        self.trades, self.local, self.remote = [], [], []
        self.tables, self.pools = {}, {}
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
            self.finish(var, self.send(var, self.start(var)))

    def start(self, var):
        """Send one variable's halos on their way: the pack, the halos a
        neighbor on this rank hands over unpacked at once, and every message
        copied to the host beside the kernels. What follows in a flow can
        run meanwhile: only the halos behind a remote face are stale until
        finish, and send goes between."""
        pack, unpackLocal, unpackRemote = self._tables(var)
        (sendPool, sendHost, sendTo), (recvPool, recvHost, recvFrom) = self._pools(var)
        # what arrives lands in the mirror before anything is sent: one
        # message per partner, under one tag, in the order the variables go
        recvs = [
            self.comm.Irecv([recvHost[part], MPIDOUBLE], source=partner, tag=0)
            for partner, part in recvFrom.items()
        ]
        # the pack turns the plane onto the neighbor's frame as it goes, so
        # nothing here has to leave the device but the messages, in one copy
        # that the kernels do not wait for
        if self.trades:
            self.pack(pack, ndim=self._ndim(var))
        if self.local:
            self.unpack(unpackLocal, ndim=self._ndim(var))
        if self.remote:
            sendPool.pullAside(sendHost)
        return recvs

    def send(self, var, recvs):
        """The messages out, once their copy has landed: each partner's is a
        slice of the mirror. Placed after a launch, the wait is for the copy
        alone while the kernels run on."""
        (sendPool, sendHost, sendTo), _ = self._pools(var)
        sends = []
        if self.remote:
            lib.pgCopyWait()
            sends = [
                self.comm.Isend([sendHost[part], MPIDOUBLE], dest=partner, tag=0)
                for partner, part in sendTo.items()
            ]
        return recvs, sends

    def finish(self, var, pending):
        """The halos in: what arrived to the device in one copy beside the
        kernels, the unpack of the remote faces once it has landed, and our
        own sends waited on. Every partner's message goes under one tag;
        messages from one source in one tag arrive in order, and this
        variable's has been received here before the next variable's is
        sent, so waiting on our own sends is all the next exchange needs."""
        recvs, sends = pending
        pack, unpackLocal, unpackRemote = self._tables(var)
        _, (recvPool, recvHost, _) = self._pools(var)
        Request.Waitall(recvs)
        if self.remote:
            recvPool.pushAside(recvHost)
            lib.pgCopyWait()
            self.unpack(unpackRemote, ndim=self._ndim(var))
        Request.Waitall(sends)

    def forget(self):
        """A block or a face re-made its arrays: every trade table and pool
        is made again."""
        self.tables, self.pools = {}, {}

    def _tables(self, var):
        """The pack and unpack tables for one variable, a trade per face. A
        neighbor on our own rank is unpacked straight out of what it packed;
        a message lands in the face's own buffer first."""
        if var not in self.tables:
            self._pools(var)
            partners = dict(self.local)
            pack, unpackLocal, unpackRemote = [], [], []
            for blk, face in self.trades:
                pack.append(Trade(blk, face, var, getattr(face, "sendBuffer_" + var)))
                if face in partners:
                    arrival = getattr(partners[face], "sendBuffer_" + var)
                    unpackLocal.append(Trade(blk, face, var, arrival))
                else:
                    arrival = getattr(face, "recvBuffer_" + var)
                    unpackRemote.append(Trade(blk, face, var, arrival))
            self.tables[var] = (
                self.backend.table(pack),
                self.backend.table(unpackLocal),
                self.backend.table(unpackRemote),
            )
        return self.tables[var]

    def _ndim(self, var):
        """How many dimensions a variable's arrays have, which its trades are
        all of; not an MPI rank."""
        return len(getattr(self.trades[0][0], var).shape)

    def _pools(self, var):
        """The remote faces' send and receive buffers for one variable, each
        a slice of one device pool, with a pinned host mirror of each pool,
        and the faces ordered so that everything for one partner rank is one
        slice of it: an exchange is one copy each way and one message per
        partner. Both sides order a partner's faces by the sending face's
        block and side, which each knows. Measured on two MI100s over the
        burner: a copy per face into host memory allocated per call, and a
        message per face, were the whole cost of the exchange."""
        if var not in self.pools:
            send, recv = "sendBuffer_" + var, "recvBuffer_" + var
            sendOrder = sorted(
                self.remote, key=lambda f: (f.commRank, f.blk.nblki, f.nface)
            )
            recvOrder = sorted(
                self.remote, key=lambda f: (f.commRank, f.neighbor, f.neighborNface)
            )
            pools = {}
            for name, order in ((send, sendOrder), (recv, recvOrder)):
                slices, offset, partners = {}, 0, {}
                for face in order:
                    n = int(np.prod(face.shapeOf(name)))
                    slices[face] = slice(offset, offset + n)
                    partners.setdefault(face.commRank, [offset, offset])[1] = offset + n
                    offset += n
                pool = self.backend.allocate((max(offset, 1),), name=f"{name}Pool")
                mirror = self.backend.pinned((max(offset, 1),))
                for face in order:
                    kind, components = face.declared[name]
                    setattr(
                        face,
                        name,
                        Array.within(
                            pool,
                            slices[face].start,
                            face.shapeOf(name),
                            name=name,
                            kind=kind,
                            components=components,
                        ),
                    )
                pools[name] = (
                    pool,
                    mirror,
                    {r: slice(a, b) for r, (a, b) in partners.items()},
                )
            self.pools[var] = (pools[send], pools[recv])
        return self.pools[var]
