"""
Trading one array's halos between blocks.

A block's halo is filled from whichever block lies across each of its block
faces, and that block may be on this rank or on another. Which block faces
trade, with whom, and how each neighbor's plane lies against ours is
settled once the blocks are wired; the exchange for one array is made from
that and the array's kind, and from then on only moves data. A face whose
neighbor is on this rank has its halo filled directly from the neighbor's
block array, in one pass and with no buffer between. A face whose
neighbor is on another rank packs its planes into a send buffer
laid out for its neighbor, carved out of that rank's pool, and its halo
costs a message and an unpack.

The direct fill, the pack and the unpack are kernels the exchange holds and the
graph launches like any other, over the block-face table and the tilings
made here. The exchange itself is the host steps between them, one method
each, in the order a graph lays them out around its launches: expect the
messages, copy them out beside the kernels once the pack is queued, send
once the copy has landed, receive, unpack, and wait on our own sends. How a
message travels -- through pinned host mirrors, or straight from the device
pools when the MPI is GPU-aware -- is the config's choice of exchange class.
"""

import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype
from mpi4py.MPI import Request

from ..backend.abi import lib
from ..backend.array import PooledArray
from ..misc import getCommRankSize


class BaseHaloExchange:
    """One array's halo exchange: its buffers on every block face met on
    another rank, its pools per other rank, its tilings over the block-face
    table, and the host steps of moving its messages."""

    # what the config calls this way of moving messages
    kind = None

    def __init__(
        self, name, faces, connOnRank, connOffRank, depth, pack, unpack, directHaloFill
    ):
        """Makes the exchange of the array :name: over the block faces of a
        table (:faces:, its entries): :connOnRank: the faces connected on
        this rank, :connOffRank: the ones connected to another; :depth: the
        planes traded, ng for a state, one for a gradient;
        :pack: and :unpack: the kernels that move the halos through the
        buffers, :directHaloFill: the one that moves a local trade's directly
        across, which only a trade within the rank can."""
        self.name, self.depth = name, depth
        self.pack, self.unpack, self.directHaloFill = pack, unpack, directHaloFill
        self.comm, self.rank, self.size = getCommRankSize()
        self.faces = faces
        # the slots on a face its buffers of this array sit in
        self.sendBuffer, self.recvBuffer = f"sendBuffer_{name}", f"recvBuffer_{name}"
        # the array's kind on any block says its components and how far
        # past a block face its trade starts
        blk = faces.entries[0].blk
        array = getattr(blk, name)
        self.components = array.components
        self.skip = array.range.exchangeStartPlane
        self.ndim = len(array.shape)
        self.backend = blk.backend
        # a face with its neighbor elsewhere has its buffers in that rank's pools
        self.ranks = sorted({f.commRank for f in connOffRank})
        self.sendPools = {
            r: self._pool(connOffRank, r, self.sendBuffer) for r in self.ranks
        }
        self.recvPools = {
            r: self._pool(connOffRank, r, self.recvBuffer) for r in self.ranks
        }
        self.recvs, self.sends = {}, {}
        # what the graph launches, over the faces each is for: the direct
        # fill over the ones connected on this rank, the pack and the
        # unpack over the ones connected to another
        self.tilings = {
            "directHaloFill": self._tiling("directHaloFill", connOnRank, self._planes),
            "pack": self._tiling(
                "pack", connOffRank, lambda f: getattr(f, self.sendBuffer).shape[:3]
            ),
            "unpack": self._tiling(
                "unpack", connOffRank, lambda f: getattr(f, self.recvBuffer).shape[:3]
            ),
        }
        self.directHaloFillArgs = dict(
            view=name, theirs=f"{name}@neighborFace", skip=self.skip, ndim=self.ndim
        )
        self.packArgs = dict(
            view=name, buffer=self.sendBuffer, skip=self.skip, ndim=self.ndim
        )
        self.unpackArgs = dict(view=name, buffer=self.recvBuffer, ndim=self.ndim)

    def _planes(self, face):
        """Gives the planes a face trades, over its block face proper: the
        planes traded, then the plane in our frame."""
        return getattr(face.blk, self.name).range.haloExtents(face.nface, self.depth)

    def _shape(self, face, theirs):
        """Gives a buffer's shape over a block face proper: the planes
        traded, the plane in our frame or, for what our neighbor reads, in
        the neighbor's."""
        depth, a, b = self._planes(face)
        plane = (b, a) if theirs and face.transposed else (a, b)
        return (depth,) + plane + self.components

    def _pool(self, connOffRank, rank, name):
        """Makes one device pool for the faces trading with :rank: and
        carves each face's buffer of this name out of it, in an order both
        ranks know: by the sending face's block and side."""
        theirs = name == self.sendBuffer
        key = (
            (lambda f: (f.blk.nblki, f.nface))
            if theirs
            else (lambda f: (f.neighbor, f.neighborNface))
        )
        faces = sorted((f for f in connOffRank if f.commRank == rank), key=key)
        shapes = [self._shape(f, theirs) for f in faces]
        offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        pool = self.backend.allocate((max(int(offsets[-1]), 1),), name=f"{name}Pool")
        for face, shape, offset in zip(faces, shapes, offsets):
            setattr(face, name, PooledArray(pool, offset, shape, name=name))
        return pool

    def _tiling(self, key, faces, planesOf):
        """Makes the tiling over the table of these faces' planes, each
        face's as :planesOf: gives them: (planes, a, b)."""
        return self.faces.tilingOver(
            (self.name, key), faces, lambda f: [((0, 0, 0), tuple(planesOf(f)))]
        )

    @property
    def messages(self):
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
    GPU-aware: no mirrors, no copies; a send waits for the pack instead,
    beside the kernels queued after it. It says on every rank where its
    pools are, and refuses one the driver does not place on the device: a
    message from anywhere else would be the staged path in disguise."""

    kind = "device"

    def __init__(self, *args):
        super().__init__(*args)
        for r in self.ranks:
            for way, pool in (("send", self.sendPools[r]), ("recv", self.recvPools[r])):
                onDevice = bool(lib.pgOnDevice(pool.ptr))
                where = "on the device" if onDevice else "not on the device"
                print(
                    f"rank {self.rank}: {self.name} {way} pool to rank {r} at {pool.ptr:#x} is {where}"
                )
                if not onDevice:
                    raise RuntimeError(
                        f"{self.name}: a device exchange's pool is {where}"
                    )

    def _memory(self, pool):
        return MPI.memory.fromaddress(pool.ptr, pool.nbytes)

    def _message(self, pool):
        """The pool's memory as a typed MPI buffer, in the case's precision."""
        return [self._memory(pool), from_numpy_dtype(pool.dtype)]

    def expect(self):
        """Posts the receives of every other rank's message, into its
        pool."""
        for r in self.ranks:
            self.recvs[r] = self.comm.Irecv(self._message(self.recvPools[r]), source=r)

    def copyOut(self):
        """Marks the pack among what the kernels have queued, for the send
        to wait on."""
        lib.pgMarkKernels()

    def send(self):
        """Posts our message to every other rank from its pool, once the
        pack is done: a message may not be posted before its buffers are
        written, and the host waits while the kernels queued after the pack
        run."""
        lib.pgCopyWait()
        for r in self.ranks:
            self.sends[r] = self.comm.Isend(self._message(self.sendPools[r]), dest=r)

    def receive(self):
        """Waits for every message, which lands in its pool."""
        Request.Waitall(list(self.recvs.values()))
