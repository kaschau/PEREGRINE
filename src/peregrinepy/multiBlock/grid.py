import numpy as np

from .topology import topology
from .gridBlock import gridBlock


class grid(topology):
    """A list of peregrinepy.multiBlock.grid objects.
    Inherits from peregrinepy.multiBlock.topology"""

    def _newBlock(self, nblki):
        return gridBlock(nblki)

    def __init__(self, nblks, ls=None):
        if ls is None:
            temp = [gridBlock(i) for i in range(nblks)]
            super().__init__(nblks, temp)
        else:
            super().__init__(nblks, ls)

    def _readBlocks(self, reader):
        """Where every block's nodes are, and so how big it is."""
        reader.readGrid(self)

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
        for blk in self:
            for face in blk.faces:
                if face.neighbor is None or face.periodicRotation is not None:
                    continue
                other = self.getBlock(face.neighbor)
                mine = np.stack(
                    [blk.array[v][face.firstPlane] for v in ("x", "y", "z")], axis=-1
                ).reshape(-1, 3)
                theirs = face.alignToMe(
                    np.stack(
                        [
                            other.array[v][other.getFace(face.neighborNface).firstPlane]
                            for v in ("x", "y", "z")
                        ],
                        axis=-1,
                    )
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
        for blk in self:
            blk.computeMetrics()

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()
