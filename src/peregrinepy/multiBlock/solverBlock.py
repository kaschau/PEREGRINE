import numpy as np

from ..backend.abi import pgDims
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

    def replace(self, name, values):
        super().replace(name, values)
        # a new array is a new arrayInfo: the tables read it again when asked
        self.mb.forget(name)

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

    def tableValue(self, name):
        """Gives what this block puts in a table under a name: an array, or
        its dims -- None for one it does not hold."""
        return getattr(self, name, None)

    @property
    def dims(self):
        return pgDims.of(self)

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
            face.setCommunication()
