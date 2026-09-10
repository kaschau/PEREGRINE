"""
Removing an interface, the inverse of a cut.

The plane an interface lies on runs on through whatever it meets, so every
pair the plane passes through merges at once, and only if the plane closes.
A pair is joined by bringing the far block into the near one's frame first,
which is why this builds on re-indexing.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .orientMixin import OrientMixin


class MergeMixin(OrientMixin):
    """Joining a grid's blocks back together, plane by plane."""

    def _pairsOnPlane(self, mb, nblki, nface):
        """Every (block, face) pair the plane of this interface passes through, or
        None if the plane does not close."""
        start = mb.getBlock(nblki).getFace(nface)
        # a periodic names is a boundary
        if start.neighbor is None or start.periodicRotation is not None:
            return None

        pairs, queue, seen = [], [(nblki, nface)], set()
        while queue:
            a, fa = queue.pop()
            A = mb.getBlock(a)
            faceA = A.getFace(fa)
            B = mb.getBlock(faceA.neighbor)
            key = frozenset({(a, fa), (B.nblki, faceA.neighborNface)})
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, fa))

            for sideA in A.faces:
                if sideA.myAxis == faceA.myAxis:
                    continue
                sideB = B.getFace(A.neighborNfaceOf(sideA.nface, across=fa))
                if sideA.neighbor is None or sideB.neighbor is None:
                    # the plane may end here, but on both sides and the same boundary
                    if (
                        sideA.neighbor is None
                        and sideB.neighbor is None
                        and sideA.bcType == sideB.bcType
                        and sideA.bcName == sideB.bcName
                    ):
                        continue
                    return None
                if (
                    sideA.periodicRotation is not None
                    or sideB.periodicRotation is not None
                ):
                    return None
                # our side neighbor must meet the same block on the far side we do
                NA = mb.getBlock(sideA.neighbor)
                planeFace = A.neighborNfaceOf(fa, across=sideA.nface)
                if NA.getFace(planeFace).neighbor != sideB.neighbor:
                    return None
                queue.append((NA.nblki, planeFace))
        return pairs

    def removablePlanes(self, mb):
        """Every plane of the grid that can be removed, as its pairs. A plane is
        reported once."""
        planes, done = [], set()
        for blk in mb:
            for face in blk.faces:
                if (blk.nblki, face.nface) in done or face.neighbor is None:
                    continue
                pairs = self._pairsOnPlane(mb, blk.nblki, face.nface)
                if pairs is None:
                    done.add((blk.nblki, face.nface))
                    continue
                for a, fa in pairs:
                    far = mb.getBlock(a).getFace(fa)
                    done.add((a, fa))
                    done.add((far.neighbor, far.neighborNface))
                planes.append(pairs)
        return planes

    def mergePlane(self, mb, pairs):
        """Merge every pair on one plane. The near block of each pair keeps its
        id; the far block is removed."""
        merged = {}
        for a, fa in pairs:
            A = mb.getBlock(a)
            faceA = A.getFace(fa)
            B = mb.getBlock(faceA.neighbor)
            self.reorientBlock(mb, B, *faceA.alignsNeighborBy)

            axis = faceA.myAxis
            lower, upper = (B, A) if faceA.amILow else (A, B)
            dropShared = [slice(None)] * 3
            dropShared[axis] = slice(1, None)
            joined = {
                name: np.concatenate(
                    [lower.array[name], upper.array[name][tuple(dropShared)]], axis=axis
                )
                for name in ("x", "y", "z")
            }
            dims = [A.ni, A.nj, A.nk]
            dims[axis] += (B.ni, B.nj, B.nk)[axis] - 1
            A.setExtents(*dims)
            for name, values in joined.items():
                A.array[name][:] = values

            # B is now in our frame, so its far face is ours on that side
            A.faces[fa - 1] = B.getFace(fa)
            merged[B.nblki] = A.nblki

        for blk in mb:
            for face in blk.faces:
                if face.neighbor in merged:
                    face.neighbor = merged[face.neighbor]
        self._compact(mb, set(merged))

    def _compact(self, mb, gone):
        """Drop the merged away blocks and number what is left from zero."""
        keep = [blk for blk in mb if blk.nblki not in gone]
        renumber = {blk.nblki: n for n, blk in enumerate(keep)}
        for blk in keep:
            blk.nblki = renumber[blk.nblki]
            # a merged block is its own base again
            blk.baseNblki, blk.baseSlice = blk.nblki, None
            for face in blk.faces:
                if face.neighbor is not None:
                    face.neighbor = renumber[face.neighbor]
        mb.data = keep
        mb.totalBlocks = len(keep)

    def matchInterfaces(self, mb):
        """Make every interface agree to the last digit.

        Two blocks that share a face each keep their own copy of the nodes on
        it, and a mesher or a translator leaves the two a rounding apart. A
        node on a block edge is shared by more than two blocks, so the copies
        are collected into groups first and each group is averaged once --
        averaging pair by pair would have each interface undo the last one.
        Returns the largest disagreement found, which says how much was being
        papered over.

        A periodic pair is left alone: its two faces are a transform apart
        rather than on top of each other, so they are not the same node.
        """
        # a compact id per node that lies on an interface, and where it lives
        ids, where = {}, []

        def idOf(blk, flat):
            key = (blk.nblki, int(flat))
            if key not in ids:
                ids[key] = len(where)
                where.append(key)
            return ids[key]

        links, seen = [], set()
        for blk in mb:
            flatOf = np.arange(blk.array["x"].size).reshape(blk.array["x"].shape)
            for face in blk.faces:
                if face.neighbor is None or face.periodicRotation is not None:
                    continue
                key = frozenset(
                    {(blk.nblki, face.nface), (face.neighbor, face.neighborNface)}
                )
                if key in seen:
                    continue
                seen.add(key)

                other = mb.getBlock(face.neighbor)
                theirFace = other.getFace(face.neighborNface)
                theirFlat = np.arange(other.array["x"].size).reshape(
                    other.array["x"].shape
                )
                mine = flatOf[face.firstPlane].ravel()
                theirs = face.alignToMe(theirFlat[theirFace.firstPlane]).ravel()
                for a, b in zip(mine, theirs):
                    links.append((idOf(blk, a), idOf(other, b)))

        if not links:
            return 0.0

        # every copy of one node in one group
        n = len(where)
        rows, cols = np.array(links).T
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, group = connected_components(graph, directed=False)

        blocks = {blk.nblki: blk for blk in mb}
        at = np.array([w[1] for w in where])
        held = np.array([w[0] for w in where])
        worst = 0.0
        for name in ("x", "y", "z"):
            values = np.array([blocks[b].array[name].flat[f] for b, f in zip(held, at)])
            total = np.bincount(group, weights=values)
            count = np.bincount(group)
            mean = (total / count)[group]
            worst = max(worst, float(np.abs(values - mean).max()))
            for m, (b, f) in enumerate(zip(held, at)):
                blocks[b].array[name].flat[f] = mean[m]
        return worst

    def mergeAll(self, mb):
        """Remove every removable plane, largest first, until none is left.
        Returns how many planes went."""
        removed = 0
        while True:
            planes = self.removablePlanes(mb)
            if not planes:
                return removed
            self.mergePlane(mb, max(planes, key=len))
            removed += 1
