#!/usr/bin/env python3
"""
A utility to cut up a peregrine grid into smaller blocks.

Current limitations: Right now we are relying on dot products to find periodic faces.
This means that we can only handle translational periodicity, and the periodic faces
must be planes.

"""

import numpy as np
import peregrinepy as pg
from verifyGrid import verify
from analyzeGrid import analyzeGrid


def faceSlice(nface):
    if nface == 1:
        return np.s_[0, :, :]
    elif nface == 2:
        return np.s_[-1, :, :]
    elif nface == 3:
        return np.s_[:, 0, :]
    elif nface == 4:
        return np.s_[:, -1, :]
    elif nface == 5:
        return np.s_[:, :, 0]
    elif nface == 6:
        return np.s_[:, :, -1]


def extractCorners(mb, incompleteBlocks, foundFaces):
    assert len(incompleteBlocks) == len(foundFaces)
    corners = {"x": [], "y": [], "z": []}
    if incompleteBlocks == []:
        return corners

    for index, nblki in enumerate(incompleteBlocks):

        blk = mb.getBlock(nblki)
        for var in corners:
            corners[var].append([])

        for found, face in zip(foundFaces[index], blk.faces):
            if found:
                faceCorners = [None, None, None, None]
                for var in corners:
                    corners[var][-1].append(faceCorners)
                continue
            for var in corners:
                array = blk.array[var]
                s_ = faceSlice(face.nface)
                faceArray = array[s_]
                faceCorners = [
                    faceArray[0, 0],
                    faceArray[0, -1],
                    faceArray[-1, 0],
                    faceArray[-1, -1],
                ]
                corners[var][-1].append(faceCorners)

    return corners


def findInteriorNeighbor(mb, incompleteBlocks, foundFaces):
    corners = extractCorners(mb, incompleteBlocks, foundFaces)
    assert all(
        [
            len(corners[var]) == len(incompleteBlocks) == len(foundFaces)
            for var in corners
        ]
    )
    if len(incompleteBlocks) == []:
        return

    for blk in mb:
        if blk.nblki not in incompleteBlocks:
            continue
        index = incompleteBlocks.index(blk.nblki)
        # Is this block complete?
        if all(foundFaces[index]):
            incompleteBlocks.pop(index)
            foundFaces.pop(index)
            for var in corners:
                corners[var].pop(index)
            continue

        # Now we go block by block, face by face and look for matching faces
        for face in blk.faces:
            if face.bcType != "b0":
                continue
            if foundFaces[index][face.nface - 1]:
                continue
            meanX = np.mean([i for i in corners["x"][index][face.nface - 1]])
            meanY = np.mean([i for i in corners["y"][index][face.nface - 1]])
            meanZ = np.mean([i for i in corners["z"][index][face.nface - 1]])
            for testIndex, nblki in enumerate(incompleteBlocks):
                testBlk = mb.getBlock(nblki)
                for testFace in testBlk.faces:
                    if foundFaces[testIndex][testFace.nface - 1]:
                        continue
                    if blk.nblki == testBlk.nblki and face.nface == testFace.nface:
                        continue
                    testMeanX = np.mean(
                        [i for i in corners["x"][testIndex][testFace.nface - 1]]
                    )
                    testMeanY = np.mean(
                        [i for i in corners["y"][testIndex][testFace.nface - 1]]
                    )
                    testMeanZ = np.mean(
                        [i for i in corners["z"][testIndex][testFace.nface - 1]]
                    )
                    dist = np.sqrt(
                        (meanX - testMeanX) ** 2
                        + (meanY - testMeanY) ** 2
                        + (meanZ - testMeanZ) ** 2
                    )
                    if dist < 1e-9:
                        face.neighbor = testBlk.nblki
                        testFace.neighbor = blk.nblki
                        foundFaces[index][face.nface - 1] = True
                        foundFaces[testIndex][testFace.nface - 1] = True

        # Is this block complete?
        if all(foundFaces[index]):
            incompleteBlocks.pop(index)
            foundFaces.pop(index)
            for var in corners:
                corners[var].pop(index)


def findPeriodicNeighbor(mb, incompleteBlocks, foundFaces):
    corners = extractCorners(mb, incompleteBlocks, foundFaces)
    assert all(
        [
            len(corners[var]) == len(incompleteBlocks) == len(foundFaces)
            for var in corners
        ]
    )
    if len(incompleteBlocks) == []:
        return

    for blk in mb:
        if blk.nblki not in incompleteBlocks:
            continue
        index = incompleteBlocks.index(blk.nblki)
        # Is this block complete?
        if all(foundFaces[index]):
            incompleteBlocks.pop(index)
            foundFaces.pop(index)
            for var in corners:
                corners[var].pop(index)
            continue

        # Now we go block by block, face by face and look for matching faces
        for face in blk.faces:
            if foundFaces[index][face.nface - 1]:
                continue
            assert face.bcType.startswith("b") and face.bcType != "b0"
            Xs = corners["x"][index][face.nface - 1]
            Ys = corners["y"][index][face.nface - 1]
            Zs = corners["z"][index][face.nface - 1]
            faceCenterX = np.mean(Xs)
            faceCenterY = np.mean(Ys)
            faceCenterZ = np.mean(Zs)

            p0 = np.array([Xs[0], Ys[0], Zs[0]])
            p1 = np.array([Xs[1], Ys[1], Zs[1]])
            p2 = np.array([Xs[3], Ys[3], Zs[3]])
            pFace = np.array([faceCenterX, faceCenterY, faceCenterZ])
            cross = np.cross(p1 - p0, p2 - p0)
            cross = cross / np.linalg.norm(cross)

            for testIndex, nblki in enumerate(incompleteBlocks):
                testBlk = mb.getBlock(nblki)
                for testFace in testBlk.faces:
                    if foundFaces[testIndex][testFace.nface - 1]:
                        continue
                    assert testFace.bcType.startswith("b") and testFace.bcType != "b0"
                    if blk.nblki == testBlk.nblki and face.nface == testFace.nface:
                        continue
                    testXs = corners["x"][testIndex][testFace.nface - 1]
                    testYs = corners["y"][testIndex][testFace.nface - 1]
                    testZs = corners["z"][testIndex][testFace.nface - 1]
                    testFaceCenterX = np.mean(testXs)
                    testFaceCenterY = np.mean(testYs)
                    testFaceCenterZ = np.mean(testZs)

                    # See if test face is in place with face so we dont accidentially pick it up
                    pTest = np.array(
                        [testFaceCenterX, testFaceCenterY, testFaceCenterZ]
                    )
                    testVector = pTest - pFace
                    testVector = testVector / np.linalg.norm(testVector)
                    dot = np.abs(np.dot(cross, testVector))
                    if dot == 1.0:
                        face.neighbor = testBlk.nblki
                        testFace.neighbor = blk.nblki
                        foundFaces[index][face.nface - 1] = True
                        foundFaces[testIndex][testFace.nface - 1] = True

        # Is this block complete?
        if all(foundFaces[index]):
            incompleteBlocks.pop(index)
            foundFaces.pop(index)
            for var in corners:
                corners[var].pop(index)


def cutBlock(mb, nblki, cutAxis, cutIndex, incompleteBlocks, foundFaces):

    oldBlk = mb.getBlock(nblki)

    # Make sure we arent trying to split at an index greater than the number of grid points
    assert (
        cutIndex < getattr(oldBlk, f"n{cutAxis}") - 1
    ), f"Error, trying to cut block {nblki} along axis {cutAxis} at index {cutIndex} >= n{cutAxis} == {getattr(oldBlk, f'n{cutAxis}')-1}."
    mb.appendBlock()
    newBlk = mb[-1]
    # Add to incomplete blocks and found faces
    incompleteBlocks.append(oldBlk.nblki)
    foundFaces.append([False for _ in range(6)])
    incompleteBlocks.append(newBlk.nblki)
    foundFaces.append([False for _ in range(6)])

    if cutAxis == "i":
        oldCutNface = 2
        newCutNface = 1
    elif cutAxis == "j":
        oldCutNface = 4
        newCutNface = 3
    elif cutAxis == "k":
        oldCutNface = 6
        newCutNface = 5

    # Before we change the cut face info copy the info
    # from the oldFace to the new block'ss opposite face
    newOppFace = newBlk.getFace(oldCutNface)
    oldOppFace = oldBlk.getFace(oldCutNface)
    newOppFace.neighbor = oldOppFace.neighbor
    newOppFace.orientation = oldOppFace.orientation
    newOppFace.bcType = oldOppFace.bcType
    newOppFace.bcFam = oldOppFace.bcFam
    foundFaces[-2][newCutNface - 1] = True
    foundFaces[-1][oldCutNface - 1] = True
    # We also need to update the oppFace neighbor of the oldBlk
    oldOppNeighbor = oldOppFace.neighbor
    if oldOppNeighbor is not None:
        neighborBlk = mb.getBlock(oldOppNeighbor)
        neighborFace = neighborBlk.getFace(oldOppFace.neighborNface)
        neighborFace.neighbor = newBlk.nblki

    # We know everything about the cut faces, so set it here
    foundFaces[-2][oldCutNface - 1] = True
    foundFaces[-1][newCutNface - 1] = True
    oldFace = oldBlk.getFace(oldCutNface)
    newFace = newBlk.getFace(newCutNface)
    oldFace.neighbor = newBlk.nblki
    newFace.neighbor = oldBlk.nblki
    oldFace.orientation = "123"
    newFace.orientation = "123"
    oldFace.bcType = "b0"
    newFace.bcType = "b0"
    oldFace.bcFam = None
    newFace.bcFam = None

    # We also can set everything for the split faces except the neighbor
    splitFaces = [
        i + 1 for i in range(6) if i + 1 != oldCutNface and i + 1 != newCutNface
    ]
    for nface in splitFaces:
        oldSplitFace = oldBlk.getFace(nface)
        newSplitFace = newBlk.getFace(nface)
        newSplitFace.orientation = oldSplitFace.orientation
        newSplitFace.bcFam = oldSplitFace.bcFam
        newSplitFace.bcType = oldSplitFace.bcType
        # If this face is a boundary, then we can add it to the found faces
        if oldSplitFace.neighbor is None:
            newSplitFace.neighbor = None
            foundFaces[-2][nface - 1] = True
            foundFaces[-1][nface - 1] = True

    # Now transfer the coordinate arrays
    if cutAxis == "i":
        oldSlice = np.s_[0 : cutIndex + 1, :, :]
        newSlice = np.s_[cutIndex::, :, :]
    elif cutAxis == "j":
        oldSlice = np.s_[:, 0 : cutIndex + 1, :]
        newSlice = np.s_[:, cutIndex::, :]
    elif cutAxis == "k":
        oldSlice = np.s_[:, :, 0 : cutIndex + 1]
        newSlice = np.s_[:, :, cutIndex::]
    for var in ["x", "y", "z"]:
        newBlk.array[var] = np.copy(oldBlk.array[var][newSlice])
        oldBlk.array[var] = np.copy(oldBlk.array[var][oldSlice])

    oldBlk.ni, oldBlk.nj, oldBlk.nk = oldBlk.array["x"].shape
    newBlk.ni, newBlk.nj, newBlk.nk = newBlk.array["x"].shape


def cutPath(mb, nblki, cutAxis):

    axisMap = {"i": 0, "j": 1, "k": 2}
    orientationMap = {"1": "i", "2": "j", "3": "k", "4": "i", "5": "j", "6": "k"}

    #              [ block, axis, switchBool ]
    blocksToCut = [[nblki, cutAxis, False]]
    blocksToCheck = [[nblki, cutAxis, False]]

    while blocksToCheck != []:
        checkBlk = mb.getBlock(blocksToCheck[0][0])
        checkAxis = blocksToCheck[0][1]
        checkSwitch = blocksToCheck[0][2]

        if checkAxis == "i":
            splitFaces = [3, 4, 5, 6]
        elif checkAxis == "j":
            splitFaces = [1, 2, 5, 6]
        elif checkAxis == "k":
            splitFaces = [1, 2, 3, 4]

        for splitFace in splitFaces:
            face = checkBlk.getFace(splitFace)
            neighbor = face.neighbor
            if neighbor is None or neighbor in [item[0] for item in blocksToCut]:
                continue
            neighborOrientation = face.neighborOrientation
            neighborOrientationIndex = neighborOrientation[axisMap[cutAxis]]
            neighborAxis = orientationMap[neighborOrientationIndex]
            if neighborOrientationIndex in ["4", "5", "6"]:
                neighborSwitch = False if checkSwitch else True
            else:
                neighborSwitch = checkSwitch

            blocksToCheck.append([neighbor, neighborAxis, neighborSwitch])
            blocksToCut.append([neighbor, neighborAxis, neighborSwitch])

        blocksToCheck.pop(0)

    return blocksToCut


def performCutOperations(mb, cutOps):
    for nblki, axis, nCuts in cutOps:
        cutBlk = mb.getBlock(nblki)
        ogNx = getattr(cutBlk, f"n{axis}")

        for cut in range(nCuts):
            blocksToCut = cutPath(mb, nblki, axis)
            cutNx = getattr(cutBlk, f"n{axis}")

            cutIndex = int(ogNx * (nCuts - cut) / (nCuts + 1))
            switchCutIndex = cutNx - cutIndex - 1

            incompleteBlocks = []
            foundFaces = []
            for cutNblki, cutAxis, switch in blocksToCut:
                assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                index = switchCutIndex if switch else cutIndex
                cutBlock(mb, cutNblki, cutAxis, index, incompleteBlocks, foundFaces)

            findInteriorNeighbor(mb, incompleteBlocks, foundFaces)
            findPeriodicNeighbor(mb, incompleteBlocks, foundFaces)
            assert incompleteBlocks == []
            assert foundFaces == []


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Cut a grid arbitrary number of times along arbitrary axis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-from",
        "--fromDir",
        action="store",
        metavar="<fromDir>",
        dest="fromDir",
        default="./from",
        help="Directory containing the gv.* and conn.yaml files to cut. Default is ./from",
        type=str,
    )
    parser.add_argument(
        "-to",
        "--toDir",
        action="store",
        metavar="<toDir>",
        dest="toDir",
        default="./to",
        help="Directory where files will be output. Default is ./to",
        type=str,
    )
    parser.add_argument(
        "-cutOps",
        "--specifyCutOps",
        action="store_true",
        dest="cutOps",
        help="".join(
            [
                "Cut grid by sequence of operations specified in ./cutOps.txt which has the format: \n",
                "# blockToCut, cutAxis, nCuts\n",
                "     0,         i,       2,\n\n",
                "The above would cut the zeroth block along the i axis twice.",
            ]
        ),
    )
    parser.add_argument(
        "-maxSize",
        "--maxBlockSize",
        action="store",
        metavar="<nCells>",
        dest="maxBlockSize",
        default=0,
        help="If this option is specified, the utility will automatically cut down the grid such that the max block size (ni*nj*nk) < targetBlockSize",
        type=int,
    )

    args = parser.parse_args()

    fromDir = args.fromDir
    toDir = args.toDir

    cutOps = args.cutOps
    maxBlockSize = args.maxBlockSize

    if cutOps and maxBlockSize > 0:
        raise ValueError("Cannot specify both cutOps and maxBlockSize")

    nblks = len([f for f in os.listdir(f"./{fromDir}") if f.endswith(".h5")])
    mb = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(mb, fromDir)
    pg.readers.readConnectivity(mb, fromDir)

    if cutOps:
        cutOperations = []
        with open("./cutOps.txt", "r") as f:
            lines = [
                ln.strip() for ln in f.readlines() if not ln.strip().startswith("#")
            ]
            for line in lines:
                ln = line.split(",")
                nblk = int(ln[0])
                axis = ln[1]
                nCuts = int(ln[2])

                cutOperations.append([nblk, axis, nCuts])

        performCutOperations(mb, cutOperations)

    elif maxBlockSize > 0:

        results = analyzeGrid(mb)
        maxCells = results["maxCells"]

        while maxCells > maxBlockSize:
            blockToCut = mb.getBlock(results["maxNblki"])
            nis = np.array(results["maxNx"])
            requiredCutsPerAxis = [0, 0, 0]
            for a in range(3):
                trialNsplits = np.ones(3, dtype=np.int32)
                trialNcells = maxCells + 1
                while trialNcells > maxBlockSize:
                    trialNsplits[a] += 1.0
                    # How many times do we need to cut this axis before block size is less than maxBlockSize?
                    trialNcells = np.product(nis / trialNsplits)
                requiredCutsPerAxis[a] = trialNsplits[a] - 1

            # We now know how many times we need to cut each block to get the max block below maxBlockSize
            # Find the cut axis that produces the minimum number of new blocks
            newBlocks = [0, 0, 0]
            axes = ["i", "j", "k"]
            for a in range(3):
                newBlocks[a] = len(cutPath(mb, blockToCut.nblki, axes[a])) * (
                    requiredCutsPerAxis[a] - 1
                )

            # Chose the cut axis with minimum
            index = newBlocks.index(min(newBlocks))
            cutOperations = [[blockToCut.nblki, axes[index], requiredCutsPerAxis[a]]]
            performCutOperations(mb, cutOperations)

            results = analyzeGrid(mb)
            maxCells = results["maxCells"]

    assert verify(mb)

    results = analyzeGrid(mb)

    maxNblki = results["maxNblki"]
    maxCells = results["maxCells"]
    minNblki = results["minNblki"]
    minCells = results["minCells"]
    mean = results["mean"]
    stdv = results["stdv"]

    ni, nj, nk = results["maxNx"]
    print(f"max block is {maxNblki} with {maxCells} cells, {ni = }, {nj = }, {nk = }.")
    ni, nj, nk = results["minNx"]
    print(f"min block is {minNblki} with {minCells} cells, {ni = }, {nj = }, {nk = }.")
    print(f"{mean = }, {stdv = }")

    pg.writers.writeGrid(mb, toDir)
    pg.writers.writeConnectivity(mb, toDir)
