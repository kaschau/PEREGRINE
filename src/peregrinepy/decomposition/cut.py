"""
Cutting a grid into smaller blocks. A cut plane through one block has to
continue through every block it meets, so a cut is a path, not a single
split; the pieces it leaves are re-paired by matching face centers.
"""

import numpy as np


def faceSlice(nface):
    """The index of the plane of nodes a face sits on."""
    return (slice(None),) * ((nface - 1) // 2) + (0 if nface % 2 else -1,)


def faceCenter(blk, face):
    """The center of a face, as the mean of its four corners."""
    center = np.empty(3)
    for n, var in enumerate(("x", "y", "z")):
        nodes = blk.array[var][faceSlice(face.nface)]
        center[n] = np.mean([nodes[0, 0], nodes[0, -1], nodes[-1, 0], nodes[-1, -1]])
    return center


def faceSearchPoint(face, center):
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


def pairCutFaces(openFaces):
    """Pair the faces a cut left open, by their centers. Every one of them
    has its partner somewhere in the set, so a face left over means the cut
    path did not close."""
    centers = [faceCenter(blk, face) for blk, face in openFaces]
    searchPoints = [
        faceSearchPoint(face, center) for (blk, face), center in zip(openFaces, centers)
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


def cutBlock(mb, nblki, cutAxis, cutIndex):
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


def cutPath(mb, nblki, cutAxis):
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


def performCutOperations(mb, cutOps):
    print("Performing cut/s...")
    for nblki, axis, nCuts in cutOps:
        cutBlk = mb.getBlock(nblki)
        ogNx = getattr(cutBlk, f"n{axis}")
        print(f"  Cutting Block {nblki}'s {axis} axis {nCuts} times.")

        for cut in range(nCuts):
            blocksToCut = cutPath(mb, nblki, axis)
            cutNx = getattr(cutBlk, f"n{axis}")

            cutIndex = int(ogNx * (nCuts - cut) / (nCuts + 1))
            switchCutIndex = cutNx - cutIndex - 1

            openFaces = []
            for cutNblki, cutAxis, switch in blocksToCut:
                assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                index = switchCutIndex if switch else cutIndex
                openFaces += cutBlock(mb, cutNblki, cutAxis, index)

            pairCutFaces(openFaces)
