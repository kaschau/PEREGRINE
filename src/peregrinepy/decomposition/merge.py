"""
Removing an interface, the inverse of a cut.

An interface between two blocks lies on a plane, and that plane runs on
through whatever it meets. Removing the interface means merging every block
pair the plane passes through, so it can only be removed if the plane closes:
followed out through the faces around each pair, it must always arrive at
another pair on the same plane, or at the same boundary on both sides, and
never at a mismatch.

Each pair becomes one block. The far block is relabelled into the near
block's frame, the coordinates are joined along the plane's axis with the
shared node plane dropped, and the near block keeps its id.
"""

import numpy as np

from .reorient import reorientBlock


def pairsOnPlane(mb, nblki, nface):
    """Every (block, face) pair the plane of this interface passes through, or
    None if the plane does not close."""
    start = mb.getBlock(nblki).getFace(nface)
    # a periodic seam is a boundary that happens to name a neighbor, not an
    # interface the grid can be merged across
    if start.neighbor is None or start.bcType.startswith("periodic"):
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
                # the plane may end here, but only if it ends on both sides
                # and on the same boundary
                if (
                    sideA.neighbor is None
                    and sideB.neighbor is None
                    and sideA.bcType == sideB.bcType
                    and sideA.bcFam == sideB.bcFam
                ):
                    continue
                return None
            if sideA.bcType.startswith("periodic") or sideB.bcType.startswith(
                "periodic"
            ):
                return None
            # the plane carries on into our side neighbor, which must meet
            # the same block on the far side that we do
            NA = mb.getBlock(sideA.neighbor)
            planeFace = A.neighborNfaceOf(fa, across=sideA.nface)
            if NA.getFace(planeFace).neighbor != sideB.neighbor:
                return None
            queue.append((NA.nblki, planeFace))
    return pairs


def removablePlanes(mb):
    """Every plane of the grid that can be removed, as its pairs. A plane is
    reported once."""
    planes, done = [], set()
    for blk in mb:
        for face in blk.faces:
            if (blk.nblki, face.nface) in done or face.neighbor is None:
                continue
            pairs = pairsOnPlane(mb, blk.nblki, face.nface)
            if pairs is None:
                done.add((blk.nblki, face.nface))
                continue
            for a, fa in pairs:
                far = mb.getBlock(a).getFace(fa)
                done.add((a, fa))
                done.add((far.neighbor, far.neighborNface))
            planes.append(pairs)
    return planes


def mergePlane(mb, pairs):
    """Merge every pair on one plane. The near block of each pair keeps its
    id; the far block is removed."""
    merged = {}
    for a, fa in pairs:
        A = mb.getBlock(a)
        faceA = A.getFace(fa)
        B = mb.getBlock(faceA.neighbor)
        reorientBlock(mb, B, *faceA.alignsNeighborBy)

        axis = faceA.myAxis
        lower, upper = (B, A) if faceA.amILow else (A, B)
        dropShared = [slice(None)] * 3
        dropShared[axis] = slice(1, None)
        for name in ("x", "y", "z"):
            A.array[name] = np.concatenate(
                [lower.array[name], upper.array[name][tuple(dropShared)]], axis=axis
            )
        dims = [A.ni, A.nj, A.nk]
        dims[axis] += (B.ni, B.nj, B.nk)[axis] - 1
        A.ni, A.nj, A.nk = dims

        # B is now in our frame, so its far face is ours on that side
        A.faces[fa - 1] = B.getFace(fa)
        merged[B.nblki] = A.nblki

    for blk in mb:
        for face in blk.faces:
            if face.neighbor in merged:
                face.neighbor = merged[face.neighbor]
    compact(mb, set(merged))


def compact(mb, gone):
    """Drop the merged away blocks and number what is left from zero."""
    keep = [blk for blk in mb if blk.nblki not in gone]
    renumber = {blk.nblki: n for n, blk in enumerate(keep)}
    for blk in keep:
        blk.nblki = renumber[blk.nblki]
        for face in blk.faces:
            if face.neighbor is not None:
                face.neighbor = renumber[face.neighbor]
    mb.data = keep
    mb.totalBlocks = len(keep)


def mergeAll(mb):
    """Remove every removable plane, largest first, until none is left.
    Returns how many planes went."""
    removed = 0
    while True:
        planes = removablePlanes(mb)
        if not planes:
            return removed
        mergePlane(mb, max(planes, key=len))
        removed += 1
