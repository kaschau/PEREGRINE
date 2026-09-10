"""
Re-indexing a block's storage.

A block's axes can be relabelled without moving the grid it holds: swap which
index runs along which direction, and reverse any of them. The coordinates,
the extents and the faces all follow, and so does every orientation string
that names this block's axes -- the block's own, and its neighbors' pointing
back at it.

The relabelling is (perm, flips): new axis m runs along old axis perm[m], and
runs backwards when flips[m].
"""

import numpy as np


def longestFirst(blk):
    """The relabelling that puts a block's longest extent on i and its second
    on j. k is reversed when the axis order is odd, so the block stays right
    handed."""
    perm = [int(a) for a in np.argsort([-blk.ni, -blk.nj, -blk.nk], kind="stable")]
    swaps = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
    return perm, [False, False, swaps % 2 == 1]


def reorientBlock(mb, blk, perm, flips):
    """Relabel blk's axes by (perm, flips), in place."""
    if perm == [0, 1, 2] and not any(flips):
        return

    face = blk.faces[0]
    newAxis = {old: new for new, old in enumerate(perm)}

    # their strings name our axes, so each character becomes our new label
    for other in mb:
        if other is blk:
            continue
        for theirs in other.faces:
            if theirs.neighbor != blk.nblki or theirs.orientation is None:
                continue
            theirs.orientation = "".join(
                face.orientationCode(
                    newAxis[ours], counterAligned != flips[newAxis[ours]]
                )
                for ours, counterAligned in map(face.signedAxis, theirs.orientation)
            )

    # ours are indexed by our axis, so permuted not rewritten; a flip inverts
    for f in blk.faces:
        if f.orientation is None:
            continue
        old = f.orientation
        f.orientation = "".join(
            face.orientationCode(
                *(lambda axis, counterAligned: (axis, counterAligned != flips[m]))(
                    *face.signedAxis(old[perm[m]])
                )
            )
            for m in range(3)
        )

    # a face keeps its identity; its number follows its axis, a flip swaps ends
    reordered = [None] * 6
    for f in blk.faces:
        m = newAxis[f.myAxis]
        low = f.amILow != flips[m]
        f.nface = 2 * m + (1 if low else 2)
        reordered[f.nface - 1] = f
    blk.faces = reordered

    dims = (blk.ni, blk.nj, blk.nk)
    blk.ni, blk.nj, blk.nk = (dims[perm[m]] for m in range(3))
    for name, values in blk.array.items():
        if values is None:
            continue
        moved = np.moveaxis(values, perm, (0, 1, 2))
        for m in range(3):
            if flips[m]:
                moved = np.flip(moved, axis=m)
        blk.array[name] = np.ascontiguousarray(moved)


def longestAxisFirst(mb):
    """Relabel every block so its longest extent is i, the axis a launch walks
    innermost."""
    for blk in mb:
        reorientBlock(mb, blk, *longestFirst(blk))
