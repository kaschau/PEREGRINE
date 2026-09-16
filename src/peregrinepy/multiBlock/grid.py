import numpy as np

from ..backend import HostBackend
from .topology import topology
from .gridBlock import gridBlock


class grid(topology):
    """A topology with coordinates: every block's nodes, and the metrics and
    halos that follow from them. A multiBlock says what every one of its
    blocks holds, each level adding its own arrays to the level below's."""

    # the halo depth of its blocks; a solver's is as deep as its widest stencil
    ng = 0

    def __init__(self):
        super().__init__()
        # where its blocks' arrays are made: numpy on the host, until a solver
        # says where its kernels run
        self.backend = HostBackend()
        # name -> (kind, components) of every array a block holds
        self.arrays = {}
        self.declareArray("nodes", kind="node", components=3)
        # cell centers are as much as a block with no solution on it can work out
        self.declareArray("cells", kind="cell", components=3)

    def declareArray(self, name, *, kind, components=()):
        """An array every block holds: its kind names the shape, the
        components what sits at each point of it."""
        if isinstance(components, int):
            components = (components,)
        self.arrays[name] = (kind, tuple(components))

    def _newBlock(self, nblki):
        return gridBlock(nblki, self)

    def detectPeriodics(self, tol=1e-8):
        """Find the interfaces that are really periodic, and how they move.

        A translator names a face by what it is joined to, not by how far away
        that is, so a periodic arrives looking like any other interface -- the
        one difference being that its nodes and its partner's do not sit on
        top of each other. The transform that carries one onto the other is
        what a periodic is, so it is read off the two directly rather than
        being declared in a file that can disagree with the grid.

        Returns how many of each kind were found.
        """
        found = {}
        for blk, face in self.faces():
            if face.neighbor is None or face.periodicRotation is not None:
                continue
            other = self.getBlock(face.neighbor)
            mine = blk.nodes.get()[face.firstPlane].reshape(-1, 3)
            theirs = face.alignToMe(
                other.nodes.get()[other.getFace(face.neighborNface).firstPlane]
            ).reshape(-1, 3)

            moved = self._transformOnto(theirs, mine, tol)
            if moved is None:
                continue
            rotation, translation = moved
            # a rotation of the identity is a face that was only carried
            turned = not np.allclose(rotation, np.eye(3), atol=tol)
            face.bcType = "periodicRot" if turned else "periodicTrans"
            face.setPeriodic(rotation=rotation, translation=translation)
            found[face.bcType] = found.get(face.bcType, 0) + 1
        return found

    @staticmethod
    def _transformOnto(theirs, mine, tol):
        """The rigid transform that carries :theirs: onto :mine:, or None if
        they are already the same points. A translation is tried first, since
        it is both the common case and exact when it is the answer."""
        delta = mine - theirs
        translation = delta.mean(axis=0)
        if np.abs(delta).max() <= tol:
            return None
        if np.abs(delta - translation).max() <= tol:
            return np.eye(3), translation

        # a rotational periodic turns about an axis through the origin, so
        # there is no translation left to find and the best rotation is the
        # orthogonal Procrustes solution
        u, _, vt = np.linalg.svd(mine.T @ theirs)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        off = np.abs(theirs @ rotation.T - mine).max()
        if off > tol:
            raise ValueError(
                "an interface is neither joined nor a rigid transform apart:"
                f" the best rotation still leaves it {off:.3e} out"
            )
        return rotation, np.zeros(3)

    def computeMetrics(self):
        for blk in self.blocks:
            blk.computeMetrics()
