"""
Re-indexing a block's storage.

A block's axes can be relabelled without moving the grid it holds: new axis m
runs along old axis perm[m], and runs backwards when flips[m]. It is how a
block being merged is brought into its partner's frame, and how a conditioned
grid gets its longest extent on i.
"""

import numpy as np


class OrientMixin:
    """Relabelling a block's axes without moving the grid it holds."""

    def _longestFirst(self, blk):
        """The relabelling that puts a block's longest extent on i and its second
        on j. k is reversed when the axis order is odd, so the block stays right
        handed."""
        perm = [int(a) for a in np.argsort([-blk.ni, -blk.nj, -blk.nk], kind="stable")]
        swaps = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
        return perm, [False, False, swaps % 2 == 1]

    def reorientBlock(self, mb, blk, perm, flips):
        """Relabel blk's axes by (perm, flips), in place."""
        if perm == [0, 1, 2] and not any(flips):
            return

        face = blk.faces[0]
        newAxis = {old: new for new, old in enumerate(perm)}

        # their strings name our axes, so each character becomes our new label
        for other in mb.blocks:
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

        # ours are indexed by our axis, so permuted not rewritten; a flip
        # inverts. A face joined to its own block is both ours and theirs, and
        # the loop above skipped it, so it is renamed here as well
        for f in blk.faces:
            if f.orientation is None:
                continue
            old, itself = f.orientation, f.neighbor == blk.nblki
            codes = []
            for m in range(3):
                axis, counterAligned = face.signedAxis(old[perm[m]])
                counterAligned = counterAligned != flips[m]
                if itself:
                    counterAligned = counterAligned != flips[newAxis[axis]]
                    axis = newAxis[axis]
                codes.append(face.orientationCode(axis, counterAligned))
            f.orientation = "".join(codes)

        # a face keeps its identity; its number follows its axis, a flip swaps ends
        reordered = [None] * 6
        for f in blk.faces:
            m = newAxis[f.myAxis]
            low = f.amILow != flips[m]
            f.nface = 2 * m + (1 if low else 2)
            reordered[f.nface - 1] = f
        blk.faces = reordered

        # the arrays, relabeled; sizing the block makes new ones, which take
        # them. Not a cell face array, which is one axis's and is made again
        # from the cells, nor a metric, remade from the nodes by computeMetrics
        dims = (blk.ni, blk.nj, blk.nk)
        moved = {}
        for name, (kind, components, rangeArgs) in blk.declared.items():
            array = getattr(blk, name)
            if (
                array is None
                or "axis" in rangeArgs
                or name == "cells"
                or name in mb.metrics
            ):
                continue
            values = np.moveaxis(array.get(), perm, (0, 1, 2))
            for m in range(3):
                if flips[m]:
                    values = np.flip(values, axis=m)
            moved[name] = values
        blk.setExtents(*(dims[perm[m]] for m in range(3)))
        for name, values in moved.items():
            getattr(blk, name).set(values)

    def longestAxisFirst(self, mb):
        """Relabel every block so its longest extent is i, the axis a launch walks
        innermost."""
        for blk in mb.blocks:
            self.reorientBlock(mb, blk, *self._longestFirst(blk))
