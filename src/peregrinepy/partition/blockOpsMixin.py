"""
Reshaping a grid's blocks.

Cutting a block splits it in two, and a cut plane has to continue through
every block it meets, so a cut is a path and the pieces it leaves are re-paired
by matching face centers. Merging is the inverse: an interface can only be
removed if the plane it lies on closes, and every pair the plane passes through
merges at once. Re-indexing relabels a block's axes without moving the grid it
holds, which is how a merged block is brought into its partner's frame and how
a conditioned grid gets its longest extent on i.

Every partitioner needs all of it -- a grid has to be cut before it can be
balanced past what whole blocks allow -- so it is mixed into BasePartitioner
rather than living on any one kind.
"""

import numpy as np


class BlockOpsMixin:
    ###########################################################################
    # Cutting a grid into smaller blocks. A cut plane through one block has to
    # continue through every block it meets, so a cut is a path, not a single
    # split; the pieces it leaves are re-paired by matching face centers.
    ###########################################################################
    def _faceSlice(self, nface):
        """The index of the plane of nodes a face sits on."""
        return (slice(None),) * ((nface - 1) // 2) + (0 if nface % 2 else -1,)

    def _faceCenter(self, blk, face):
        """The center of a face, as the mean of its four corners."""
        center = np.empty(3)
        for n, var in enumerate(("x", "y", "z")):
            nodes = blk.array[var][self._faceSlice(face.nface)]
            center[n] = np.mean(
                [nodes[0, 0], nodes[0, -1], nodes[-1, 0], nodes[-1, -1]]
            )
        return center

    def _faceSearchPoint(self, face, center):
        """Where to go looking for a face's partner. Only a periodic sits
        somewhere other than on top of its partner, so only a periodic moves."""
        if face.bcType == "periodicTransLow":
            return center + face.periodicAxis * face.periodicSpan
        elif face.bcType == "periodicTransHigh":
            return center - face.periodicAxis * face.periodicSpan
        elif face.bcType == "periodicRotLow":
            return np.matmul(face.array["periodicRotMatrixUp"], center)
        elif face.bcType == "periodicRotHigh":
            return np.matmul(face.array["periodicRotMatrixDown"], center)
        return center

    def _pairCutFaces(self, openFaces):
        """Pair the faces a cut left open, by their centers. Every one of them
        has its partner somewhere in the set, so a face left over means the cut
        path did not close."""
        centers = [self._faceCenter(blk, face) for blk, face in openFaces]
        searchPoints = [
            self._faceSearchPoint(face, center)
            for (blk, face), center in zip(openFaces, centers)
        ]

        for index, (blk, face) in enumerate(openFaces):
            # an earlier face may already have claimed us
            if face.neighbor is not None:
                continue
            for testIndex, (testBlk, testFace) in enumerate(openFaces):
                if testIndex == index or testFace.neighbor is not None:
                    continue
                dist = np.linalg.norm(searchPoints[index] - centers[testIndex])
                if dist < 1e-9:
                    face.neighbor = testBlk.nblki
                    testFace.neighbor = blk.nblki
                    break
            else:
                raise ValueError(
                    f"Block {blk.nblki} face {face.nface} has nothing to pair with across the cut."
                )

    def cutBlock(self, mb, nblki, cutAxis, cutIndex):
        """Split a block in two at cutIndex along cutAxis. The low half stays as
        the block, the high half is appended to the multiBlock. Returns the
        (block, face) pairs the cut left open, to be paired up once every block
        on the path has been cut."""
        oldBlk = mb.getBlock(nblki)
        axis = "ijk".index(cutAxis)

        # Make sure we arent trying to split at an index greater than the number of grid points
        assert (
            cutIndex < getattr(oldBlk, f"n{cutAxis}") - 1
        ), f"Error, trying to cut block {nblki} along axis {cutAxis} at index {cutIndex} >= n{cutAxis} == {getattr(oldBlk, f'n{cutAxis}')-1}."
        mb.appendBlock()
        newBlk = mb[-1]

        newCutNface = 2 * axis + 1
        oldCutNface = 2 * axis + 2

        # Before we change the cut face info copy the info
        # from the oldFace to the new block's opposite face
        oldCutFace = oldBlk.getFace(oldCutNface)
        newOppFace = newBlk.getFace(oldCutNface)
        newOppFace.neighbor = oldCutFace.neighbor
        newOppFace.orientation = oldCutFace.orientation
        newOppFace.bcType = oldCutFace.bcType
        newOppFace.bcFam = oldCutFace.bcFam
        if oldCutFace.bcType.startswith("periodic"):
            newOppFace.periodicSpan = oldCutFace.periodicSpan
            newOppFace.periodicAxis = oldCutFace.periodicAxis

        # We also need to update the oppFace neighbor of the oldBlk
        if oldCutFace.neighbor is not None:
            neighborBlk = mb.getBlock(oldCutFace.neighbor)
            neighborBlk.getFace(oldCutFace.neighborNface).neighbor = newBlk.nblki

        # We know everything about the cut faces, so set it here
        newCutFace = newBlk.getFace(newCutNface)
        oldCutFace.neighbor = newBlk.nblki
        newCutFace.neighbor = oldBlk.nblki
        oldCutFace.orientation = "123"
        newCutFace.orientation = "123"
        oldCutFace.bcType = "interior"
        newCutFace.bcType = "interior"
        oldCutFace.bcFam = None
        newCutFace.bcFam = None

        # the four faces along the cut split in two; only the neighbor is unknown
        openFaces = []
        for nface in (n for n in range(1, 7) if (n - 1) // 2 != axis):
            oldSplitFace = oldBlk.getFace(nface)
            newSplitFace = newBlk.getFace(nface)
            newSplitFace.orientation = oldSplitFace.orientation
            newSplitFace.bcFam = oldSplitFace.bcFam
            newSplitFace.bcType = oldSplitFace.bcType
            # if the split face is a periodic, they need the perodic info
            if oldSplitFace.bcType.startswith("periodic"):
                newSplitFace.periodicSpan = oldSplitFace.periodicSpan
                newSplitFace.periodicAxis = oldSplitFace.periodicAxis
            if oldSplitFace.neighbor is None:
                newSplitFace.neighbor = None
                continue
            # our neighbor is cut too, so which halves meet waits for the path
            oldSplitFace.neighbor = None
            openFaces.append((oldBlk, oldSplitFace))
            openFaces.append((newBlk, newSplitFace))

        # cutIndex is local, so it lands that far along whatever slab we already are
        base = list(
            oldBlk.baseSlice or (0, oldBlk.ni - 1, 0, oldBlk.nj - 1, 0, oldBlk.nk - 1)
        )
        low, high = list(base), list(base)
        low[2 * axis + 1] = base[2 * axis] + cutIndex
        high[2 * axis] = base[2 * axis] + cutIndex
        newBlk.baseNblki = oldBlk.baseNblki
        oldBlk.baseSlice, newBlk.baseSlice = tuple(low), tuple(high)

        # Now transfer the coordinate arrays
        oldSlice, newSlice = [slice(None)] * 3, [slice(None)] * 3
        oldSlice[axis] = slice(0, cutIndex + 1)
        newSlice[axis] = slice(cutIndex, None)
        for var in ["x", "y", "z"]:
            newBlk.array[var] = np.copy(oldBlk.array[var][tuple(newSlice)])
            oldBlk.array[var] = np.copy(oldBlk.array[var][tuple(oldSlice)])

        oldBlk.ni, oldBlk.nj, oldBlk.nk = oldBlk.array["x"].shape
        newBlk.ni, newBlk.nj, newBlk.nk = newBlk.array["x"].shape

        return openFaces

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

                openFaces = []
                for cutNblki, cutAxis, switch in blocksToCut:
                    assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                    index = switchCutIndex if switch else cutIndex
                    openFaces += self.cutBlock(mb, cutNblki, cutAxis, index)

                self._pairCutFaces(openFaces)

    ###########################################################################
    # Removing an interface, the inverse of a cut. The plane an interface lies on
    # runs on through whatever it meets, so every pair it passes through merges
    # at once, and only if the plane closes.
    ###########################################################################
    def _pairsOnPlane(self, mb, nblki, nface):
        """Every (block, face) pair the plane of this interface passes through, or
        None if the plane does not close."""
        start = mb.getBlock(nblki).getFace(nface)
        # a periodic names is a boundary
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
                    # the plane may end here, but on both sides and the same boundary
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

    ###########################################################################
    # Re-indexing a block's storage. A block's axes can be relabelled without
    # moving the grid it holds: new axis m runs along old axis perm[m], and runs
    # backwards when flips[m].
    ###########################################################################
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

    def longestAxisFirst(self, mb):
        """Relabel every block so its longest extent is i, the axis a launch walks
        innermost."""
        for blk in mb:
            self.reorientBlock(mb, blk, *self._longestFirst(blk))
