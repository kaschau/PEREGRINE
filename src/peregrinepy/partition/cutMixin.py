"""
Cutting a grid into smaller blocks.

A cut plane through one block has to continue through every block it meets, so
a cut is a path and not a single split. Every block on that path is cut, and
the faces the cut leaves open are paired afterwards: the neighbor was cut too,
so which of its halves a face now meets is which way its cut axis runs in the
neighbor's frame, which the path already worked out.
"""

import numpy as np


class CutMixin:
    """Splitting a grid's blocks, and the table that says what came from where."""

    @staticmethod
    def _pairCutPieces(pieces, pending):
        """Point the faces a cut left open at the right half of their old
        neighbor. The cut ran on through that neighbor too, so which of its
        halves we now meet is just which way our cut axis runs in its frame."""
        for lowFace, highFace, neighbor, counterAligned in pending:
            low, high = pieces[neighbor]
            lowFace.neighbor, highFace.neighbor = (
                (high, low) if counterAligned else (low, high)
            )

    def cutBlock(self, mb, nblki, cutAxis, cutIndex):
        """Split a block in two at cutIndex along cutAxis. The low half stays as
        the block, the high half is appended to the multiBlock. Returns the
        (block, face) pairs the cut left open, to be paired up once every block
        on the path has been cut."""
        oldBlk = mb.getBlock(nblki)
        axis = "ijk".index(cutAxis)
        nNodes = (oldBlk.ni, oldBlk.nj, oldBlk.nk)[axis]

        # every piece needs a cell, so the last place a cut can land is one
        # short of the far end
        assert cutIndex < nNodes - 1, (
            f"cannot cut block {nblki} along {cutAxis} at {cutIndex}:"
            f" n{cutAxis} is {nNodes}, so the last cut is at {nNodes - 2}"
        )
        mb.appendBlock()
        newBlk = mb[-1]

        newCutNface = 2 * axis + 1
        oldCutNface = 2 * axis + 2

        # Before we change the cut face info copy the info
        # from the oldFace to the new block's opposite face
        oldCutFace = oldBlk.getFace(oldCutNface)
        newOppFace = newBlk.getFace(oldCutNface)
        newOppFace.copyFrom(oldCutFace)
        newOppFace.neighbor = oldCutFace.neighbor

        # We also need to update the oppFace neighbor of the oldBlk
        if oldCutFace.neighbor is not None:
            neighborBlk = mb.getBlock(oldCutFace.neighbor)
            neighborBlk.getFace(oldCutFace.neighborNface).neighbor = newBlk.nblki

        # We know everything about the cut faces, so set it here
        # a cut face may have met a partner across the grid before it met its
        # own other half, and none of that survives the cut
        newCutFace = newBlk.getFace(newCutNface)
        oldCutFace.setInterior(newBlk.nblki)
        newCutFace.setInterior(oldBlk.nblki)

        # the four faces along the cut split in two; only the neighbor is unknown
        pending = []
        for nface in (n for n in range(1, 7) if (n - 1) // 2 != axis):
            oldSplitFace = oldBlk.getFace(nface)
            newSplitFace = newBlk.getFace(nface)
            newSplitFace.copyFrom(oldSplitFace)
            if oldSplitFace.neighbor is None:
                newSplitFace.neighbor = None
                continue
            # our neighbor is cut too, so which halves meet waits for the path
            _, counterAligned = oldSplitFace.signedAxis(oldSplitFace.orientation[axis])
            pending.append(
                (oldSplitFace, newSplitFace, oldSplitFace.neighbor, counterAligned)
            )

        # cutIndex is local, so it lands that far along whatever slab we already are
        base = list(
            oldBlk.baseSlice or (0, oldBlk.ni - 1, 0, oldBlk.nj - 1, 0, oldBlk.nk - 1)
        )
        low, high = list(base), list(base)
        low[2 * axis + 1] = base[2 * axis] + cutIndex
        high[2 * axis] = base[2 * axis] + cutIndex
        newBlk.baseNblki = oldBlk.baseNblki
        oldBlk.baseSlice, newBlk.baseSlice = tuple(low), tuple(high)

        # Each half is a smaller block, so it is resized before being filled and
        # everything it derives from its extents comes back the right shape
        # rather than the old one. Whatever a kind of block holds along its
        # extents has to be taken before either of them is resized.
        halves = oldBlk.splitAlong(axis, cutIndex)
        lowDims = [oldBlk.ni, oldBlk.nj, oldBlk.nk]
        highDims = list(lowDims)
        lowDims[axis] = cutIndex + 1
        highDims[axis] -= cutIndex
        oldBlk.setExtents(*lowDims)
        newBlk.setExtents(*highDims)
        for var, (lowHalf, highHalf) in halves.items():
            oldBlk.array[var][:] = lowHalf
            newBlk.array[var][:] = highHalf

        return pending

    @staticmethod
    def cutTable(mb):
        """Which base block each block is a piece of, and which slab of it, as a
        (nblks, 7) table of baseNblki and inclusive node bounds i0, i1, j0, j1,
        k0, k1. A decomposition is this table plus which rank owns each row."""
        table = np.empty((len(mb), 7), dtype=np.int32)
        for n, blk in enumerate(mb):
            table[n, 0] = blk.baseNblki
            table[n, 1:] = blk.baseSlice or (
                0,
                blk.ni - 1,
                0,
                blk.nj - 1,
                0,
                blk.nk - 1,
            )
        return table

    def cutPath(self, mb, nblki, cutAxis):
        """Every block a cut runs on into, as [block, the axis it is cut on,
        whether its cut index counts from the far end]."""
        #              [ block, axis, switchBool ]
        blocksToCut = [[nblki, cutAxis, False]]
        blocksToCheck = [[nblki, cutAxis, False]]

        while blocksToCheck != []:
            checkNblki, checkAxis, checkSwitch = blocksToCheck.pop(0)
            checkBlk = mb.getBlock(checkNblki)
            checkAxisIndex = "ijk".index(checkAxis)

            # the cut runs out through the four faces it is parallel to
            for splitFace in (n for n in range(1, 7) if (n - 1) // 2 != checkAxisIndex):
                face = checkBlk.getFace(splitFace)
                neighbor = face.neighbor
                if neighbor is None or neighbor in [item[0] for item in blocksToCut]:
                    continue
                axis, counterAligned = face.signedAxis(face.orientation[checkAxisIndex])
                neighborSwitch = checkSwitch != counterAligned
                blocksToCheck.append([neighbor, "ijk"[axis], neighborSwitch])
                blocksToCut.append([neighbor, "ijk"[axis], neighborSwitch])

        return blocksToCut

    def performCutOperations(self, mb, cutOps):
        print("Performing cut/s...")
        for nblki, axis, nCuts in cutOps:
            cutBlk = mb.getBlock(nblki)
            ogNx = getattr(cutBlk, f"n{axis}")
            print(f"  Cutting Block {nblki}'s {axis} axis {nCuts} times.")

            for cut in range(nCuts):
                blocksToCut = self.cutPath(mb, nblki, axis)
                cutNx = getattr(cutBlk, f"n{axis}")

                cutIndex = int(ogNx * (nCuts - cut) / (nCuts + 1))
                switchCutIndex = cutNx - cutIndex - 1

                # which two blocks each block on the path became
                pieces, pending = {}, []
                for cutNblki, cutAxis, switch in blocksToCut:
                    assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                    index = switchCutIndex if switch else cutIndex
                    pending += self.cutBlock(mb, cutNblki, cutAxis, index)
                    pieces[cutNblki] = (cutNblki, mb[-1].nblki)

                self._pairCutPieces(pieces, pending)
