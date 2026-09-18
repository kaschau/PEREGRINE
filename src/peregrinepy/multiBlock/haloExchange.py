"""
Trading one array's halos between blocks.

A block's halo is filled from whichever block lies across each of its block
faces, and that block may be on this rank or on another. Which block faces
trade, with whom, and how each neighbor's plane lies against ours is
settled once the blocks are wired; the exchange for one array is made from
that and the array's kind, and from then on only moves data. Every trading
face packs its planes into a send buffer laid out for its neighbor. A face
whose neighbor is on this rank holds the neighbor's send buffer as its own
receive buffer, so its halo is unpacked straight out of what the neighbor
packed, on the device. A face whose neighbor is on another rank has its
buffers carved out of that rank's pools, and its halo costs a message.

The pack and the two unpacks are kernels the exchange holds and the graph
launches like any other, over the block-face table and the tilings made
here. The exchange itself is the host steps between them, one method each, in the order a
graph lays them out around its launches: expect the messages, copy them
out beside the kernels once the pack is queued, send once the copy has
landed, receive, unpack, and wait on our own sends. How a message travels
-- through pinned host mirrors, or straight from the device pools when the
MPI is GPU-aware -- is the config's choice of exchange class.
"""

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Request

from ..backend.abi import lib
from ..backend.array import PooledArray
from ..misc import getCommRankSize


class BaseHaloExchange:
    """One array's halo exchange: its buffers on every trading block face,
    its pools per other rank, its tilings over the block-face table, and
    the host steps of moving its messages."""

    # what the config calls this way of moving messages
    kind = None

    def __init__(self, name, faces, blockFacesBy, depth, pack, unpack):
        """Makes the exchange of the array :name: over the block faces of a
        table (:faces:, its entries), sorted in :blockFacesBy:: the ones
        that trade, the pairs of a face and the face it meets on this rank
        (local), the ones whose neighbor is on another rank (remote);
        :depth: the planes traded, ng for a state, one for a gradient;
        :pack: and :unpack: the kernels that move the halos through the
        buffers."""
        self.name, self.depth = name, depth
        self.pack, self.unpack = pack, unpack
        self.comm, self.rank, self.size = getCommRankSize()
        self.faces = faces
        trading, local, remote = (
            blockFacesBy[k] for k in ("trading", "local", "remote")
        )
        self.send, self.recv = f"sendBuffer_{name}", f"recvBuffer_{name}"
        # the array's kind on any block says its components and how far
        # past a block face its trade starts
        blk = trading[0].blk if trading else faces.entries[0].blk
        array = getattr(blk, name)
        self.components = array.components
        self.skip = array.range.exchangeStartPlane
        self.ndim = len(array.shape)
        self.backend = blk.backend
        # a face with its neighbor here packs into a buffer of its own and
        # unpacks out of the neighbor's; the slots are the face's, declared
        for face, theirs in local:
            setattr(face, self.send, self._buffer(face, self.send, theirs=True))
        for face, theirs in local:
            setattr(face, self.recv, getattr(theirs, self.send))
        # a face with its neighbor elsewhere has its buffers in that rank's pools
        self.ranks = sorted({f.commRank for f in remote})
        self.sendPools = {r: self._pool(remote, r, self.send) for r in self.ranks}
        self.recvPools = {r: self._pool(remote, r, self.recv) for r in self.ranks}
        self.recvs, self.sends = {}, {}
        # what the graph launches: the pack over every trading face, the
        # unpack over the faces met here, and over the faces met elsewhere
        self.tilings = {
            "pack": self._tiling("pack", trading, self.send),
            "local": self._tiling("local", [f for f, _ in local], self.recv),
            "remote": self._tiling("remote", remote, self.recv),
        }
        self.packArgs = dict(
            view=name, buffer=self.send, skip=self.skip, ndim=self.ndim
        )
        self.unpackArgs = dict(view=name, buffer=self.recv, ndim=self.ndim)

    def _shape(self, face, theirs):
        """Gives a buffer's shape over a block face proper: the planes
        traded, the plane in our frame or, for what our neighbor reads, in
        the neighbor's."""
        depth, a, b = getattr(face.blk, self.name).range.haloExtents(
            face.nface, self.depth
        )
        plane = (b, a) if theirs and face._transposed else (a, b)
        return (depth,) + plane + self.components

    def _buffer(self, face, name, theirs):
        return self.backend.allocate(self._shape(face, theirs), name=name)

    def _pool(self, remote, rank, name):
        """Makes one device pool for the faces trading with :rank: and
        carves each face's buffer of this name out of it, in an order both
        ranks know: by the sending face's block and side."""
        theirs = name == self.send
        key = (
            (lambda f: (f.blk.nblki, f.nface))
            if theirs
            else (lambda f: (f.neighbor, f.neighborNface))
        )
        faces = sorted((f for f in remote if f.commRank == rank), key=key)
        shapes = [self._shape(f, theirs) for f in faces]
        offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        pool = self.backend.allocate((max(int(offsets[-1]), 1),), name=f"{name}Pool")
        for face, shape, offset in zip(faces, shapes, offsets):
            setattr(face, name, PooledArray(pool, offset, shape, name=name))
        return pool

    def _tiling(self, key, faces, buffer):
        """Makes the tiling of these faces' buffer planes over the table."""
        return self.faces.tilingOver(
            (self.name, key),
            faces,
            lambda f: [((0, 0, 0), getattr(f, buffer).shape[:3])],
        )

    @property
    def remote(self):
        """Says whether any message travels: a step then goes through the
        host and ends a captured graph."""
        return bool(self.ranks)

    def expect(self):
        """Posts the receives of every other rank's message."""
        raise NotImplementedError

    def copyOut(self):
        """Readies the packed send buffers to be sent, behind the kernels
        queued so far."""
        raise NotImplementedError

    def send(self):
        """Posts our message to every other rank once its buffers are
        ready."""
        raise NotImplementedError

    def receive(self):
        """Waits for every message and puts it where the unpack reads
        it."""
        raise NotImplementedError

    def sent(self):
        """Waits for our messages to be taken, so the buffers may be packed
        again."""
        Request.Waitall(list(self.sends.values()))


class HostStagedHaloExchange(BaseHaloExchange):
    """Messages staged through pinned host mirrors of the pools: one copy
    out beside the kernels, one message per rank each way, one copy in."""

    kind = "hostStaged"

    def __init__(self, *args):
        super().__init__(*args)
        self.sendMirrors = {
            r: self.backend.pinned(self.sendPools[r].shape) for r in self.ranks
        }
        self.recvMirrors = {
            r: self.backend.pinned(self.recvPools[r].shape) for r in self.ranks
        }

    def expect(self):
        """Posts the receives of every other rank's message, into its
        mirror."""
        for r in self.ranks:
            self.recvs[r] = self.comm.Irecv([self.recvMirrors[r], MPI.DOUBLE], source=r)

    def copyOut(self):
        """Copies every rank's packed send pool to its host mirror, beside
        the kernels, after what they have queued."""
        for r in self.ranks:
            self.sendPools[r].pullAside(self.sendMirrors[r])

    def send(self):
        """Posts our message to every other rank once the copy to its
        mirror has landed."""
        lib.pgCopyWait()
        for r in self.ranks:
            self.sends[r] = self.comm.Isend([self.sendMirrors[r], MPI.DOUBLE], dest=r)

    def receive(self):
        """Waits for every message and copies it from its mirror to the
        device beside the kernels; the unpack may be launched once this
        returns."""
        Request.Waitall(list(self.recvs.values()))
        for r in self.ranks:
            self.recvPools[r].pushAside(self.recvMirrors[r])
        lib.pgCopyWait()


class DeviceHaloExchange(BaseHaloExchange):
    """Messages straight from and into the device pools, for an MPI that is
    GPU-aware: no mirrors, no copies; a send waits for the pack instead."""

    kind = "device"

    def _memory(self, pool):
        return MPI.memory.fromaddress(pool.ptr, pool.nbytes)

    def expect(self):
        """Posts the receives of every other rank's message, into its
        pool."""
        for r in self.ranks:
            self.recvs[r] = self.comm.Irecv(
                [self._memory(self.recvPools[r]), MPI.DOUBLE], source=r
            )

    def copyOut(self):
        """Waits for the pack to finish: a message may not be posted before
        its buffers are written."""
        lib.pgFence()

    def send(self):
        """Posts our message to every other rank from its pool."""
        for r in self.ranks:
            self.sends[r] = self.comm.Isend(
                [self._memory(self.sendPools[r]), MPI.DOUBLE], dest=r
            )

    def receive(self):
        """Waits for every message, which lands in its pool."""
        Request.Waitall(list(self.recvs.values()))
