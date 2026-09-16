import numpy as np

from ..abi import pgDims
from .haloMixin import HaloMixin
from .restartBlock import restartBlock
from .solverMetricsMixin import SolverMetricsMixin
from .solverFace import solverFace


class solverBlock(restartBlock, SolverMetricsMixin, HaloMixin):
    def __init__(self, nblki, mb):
        # the solver this block belongs to
        self.mb = mb
        restartBlock.__init__(self, nblki, mb)
        self.ne = 5 + self.ns - 1

    def setExtents(self, ni, nj, nk):
        """A face is shaped by the block it bounds, so it learns how big that
        block is at the same moment the block does."""
        for face in self.faces:
            face.setExtents(ni, nj, nk, self.ne)
        super().setExtents(ni, nj, nk)
        # every column and range read off this block was of the old shape
        self.mb.table.forgetAll()
        self.mb.haloExchange.forget()
        self.mb.facesChanged = True

    def replace(self, name, values):
        super().replace(name, values)
        # a new array is a new record: the table reads it again when asked
        self.mb.table.forget(name)

    def primitives(self):
        """The primitive vector of every cell, p, u, v, w, T, Y(0 .. ns - 2),
        as a host array derived from the state: p and T are q's, the rest
        Q's over its density."""
        Q, q = self.Q.get(), self.q.get()
        prims = np.empty(Q.shape, dtype=Q.dtype, order="F")
        rhoinv = 1.0 / Q[..., 0]
        prims[..., 0] = q[..., 0]
        prims[..., 1:4] = Q[..., 1:4] * rhoinv[..., None]
        prims[..., 4] = q[..., 1]
        prims[..., 5:] = Q[..., 5:] * rhoinv[..., None]
        return prims

    def column(self, name):
        """What a table takes from this block: its shape, or an array by
        name -- None for one it does not hold."""
        if name == "dims":
            return pgDims.of(self)
        return getattr(self, name, None)

    def _newFace(self, nface):
        return solverFace(nface, self.ng, self)

    @property
    def interior(self):
        """The slice of this block's arrays that is not halo."""
        ng = self.ng
        return np.s_[ng:-ng, ng:-ng, ng:-ng]

    def setBlockCommunication(self):
        for face in self.faces:
            if face.neighbor is None:
                continue
            face.setCommunication(self.nblki)
