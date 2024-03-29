#!/usr/bin/env python3
"""
A utility to cut up a peregrine grid into smaller blocks.

"""

import numpy as np
import peregrinepy as pg
from verifyGrid import verify
from analyzeGrid import analyzeGrid
from lxml import etree


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
    if len(incompleteBlocks) == []:
        return

    corners = extractCorners(mb, incompleteBlocks, foundFaces)
    assert all(
        [
            len(corners[var]) == len(incompleteBlocks) == len(foundFaces)
            for var in corners
        ]
    )

    for index, nblki in enumerate(incompleteBlocks):
        blk = mb.getBlock(nblki)
        # Is this block complete?
        if all(foundFaces[index]):
            continue

        # Now we go block by block, face by face and look for matching faces
        for face in blk.faces:
            if face.bcType != "b0" and not face.bcType.startswith("periodic"):
                continue
            if foundFaces[index][face.nface - 1]:
                continue
            meanX = np.mean([i for i in corners["x"][index][face.nface - 1]])
            meanY = np.mean([i for i in corners["y"][index][face.nface - 1]])
            meanZ = np.mean([i for i in corners["z"][index][face.nface - 1]])
            # Translate periodics
            if face.bcType == "periodicTransLow":
                meanX += face.periodicAxis[0] * face.periodicSpan
                meanY += face.periodicAxis[1] * face.periodicSpan
                meanZ += face.periodicAxis[2] * face.periodicSpan
            elif face.bcType == "periodicTransHigh":
                meanX -= face.periodicAxis[0] * face.periodicSpan
                meanY -= face.periodicAxis[1] * face.periodicSpan
                meanZ -= face.periodicAxis[2] * face.periodicSpan
            elif face.bcType == "periodicRotLow":
                mean = np.array([meanX, meanY, meanZ])
                points = np.matmul(face.array["periodicRotMatrixUp"], mean)
                meanX = points[0]
                meanY = points[1]
                meanZ = points[2]
            elif face.bcType == "periodicRotHigh":
                mean = np.array([meanX, meanY, meanZ])
                points = np.matmul(face.array["periodicRotMatrixDown"], mean)
                meanX = points[0]
                meanY = points[1]
                meanZ = points[2]

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

    # Filter out all complete blocks
    completeIndex = []
    for index, found in enumerate(foundFaces):
        if all(found):
            completeIndex.append(index)
    for index in sorted(completeIndex, reverse=True):
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
    # from the oldFace to the new block's opposite face
    newOppFace = newBlk.getFace(oldCutNface)
    oldOppFace = oldBlk.getFace(oldCutNface)
    newOppFace.neighbor = oldOppFace.neighbor
    newOppFace.orientation = oldOppFace.orientation
    newOppFace.bcType = oldOppFace.bcType
    newOppFace.bcFam = oldOppFace.bcFam
    if oldOppFace.bcType.startswith("periodic"):
        newOppFace.periodicSpan = oldOppFace.periodicSpan
        newOppFace.periodicAxis = oldOppFace.periodicAxis

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
        # if the split face is a periodic, they need the perodic info
        if oldSplitFace.bcType.startswith("periodic"):
            newSplitFace.periodicSpan = oldSplitFace.periodicSpan
            newSplitFace.periodicAxis = oldSplitFace.periodicAxis
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
            orientation = face.orientation
            cutOrientation = orientation[axisMap[checkAxis]]
            if cutOrientation in ["4", "5", "6"]:
                neighborSwitch = False if checkSwitch else True
            else:
                neighborSwitch = checkSwitch

            neighborCutAxis = orientationMap[cutOrientation]
            blocksToCheck.append([neighbor, neighborCutAxis, neighborSwitch])
            blocksToCut.append([neighbor, neighborCutAxis, neighborSwitch])

        blocksToCheck.pop(0)

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

            incompleteBlocks = []
            foundFaces = []
            for cutNblki, cutAxis, switch in blocksToCut:
                assert getattr(mb.getBlock(cutNblki), f"n{cutAxis}") == cutNx
                index = switchCutIndex if switch else cutIndex
                cutBlock(mb, cutNblki, cutAxis, index, incompleteBlocks, foundFaces)

            findInteriorNeighbor(mb, incompleteBlocks, foundFaces)
            assert incompleteBlocks == []
            assert foundFaces == []


if __name__ == "__main__":
    import argparse

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
        help="Directory containing the g.* and conn.yaml files to cut. Default is ./from",
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
    parser.add_argument(
        "-bcFamPath",
        action="store",
        metavar="<bcFamPath>",
        dest="bcFamPath",
        default="./",
        help="""If your grid has periodics, we need to the periodic data from bcFams.""",
        type=str,
    )

    args = parser.parse_args()

    fromDir = args.fromDir
    toDir = args.toDir

    cutOps = args.cutOps
    maxBlockSize = args.maxBlockSize
    bcFamPath = args.bcFamPath

    if cutOps and maxBlockSize > 0:
        raise ValueError("Cannot specify both cutOps and maxBlockSize")

    tree = etree.parse(f"{fromDir}/g.xmf")
    nblks = len(tree.getroot().find("Domain").find("Grid"))
    mb = pg.multiBlock.grid(nblks)
    pg.readers.readGrid(mb, fromDir)
    pg.readers.readConnectivity(mb, fromDir)
    try:
        pg.readers.readBcs(mb, bcFamPath)
    except FileNotFoundError:
        print("No bcFam.yaml file provided, assuming no periodics.")

    if cutOps:
        cutOperations = []
        with open("./cutOps.txt", "r") as f:
            lines = [
                ln.strip() for ln in f.readlines() if not ln.strip().startswith("#")
            ]
            for line in lines:
                if line == "" or line.startswith("#"):
                    continue
                ln = [i.strip() for i in line.split(",")]
                nblk = int(ln[0])
                axis = ln[1]
                nCuts = int(ln[2])

                cutOperations.append([nblk, axis, nCuts])

        performCutOperations(mb, cutOperations)

    elif maxBlockSize > 0:
        results = analyzeGrid(mb)
        print(f"  ... max block size {results['maxCells']}")
        maxCells = results["maxCells"]

        with open("cutLog.log", "w") as f:
            while maxCells > maxBlockSize:
                blockToCut = mb.getBlock(results["maxNblki"])
                nis = np.array(results["maxNx"])
                requiredCutsPerAxis = [0, 0, 0]
                for a in range(3):
                    trialNsplits = np.ones(3, dtype=np.int32)
                    trialNcells = maxCells
                    while trialNcells > maxBlockSize:
                        trialNsplits[a] += 1
                        # How many times do we need to cut this axis before block size is less than maxBlockSize?
                        trialNcells = np.prod(nis / trialNsplits)
                        if trialNsplits[a] >= nis[a]:
                            raise ValueError(
                                "I have to split the largest block to single cell sizes to meet your requested maxBlockSize. Are you sure thats a reasonable block size?"
                            )
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
                cutOperations = [
                    [blockToCut.nblki, axes[index], requiredCutsPerAxis[a]]
                ]
                f.write(
                    f"{blockToCut.nblki}, {axes[index]}, {requiredCutsPerAxis[a]}\n"
                )

                performCutOperations(mb, cutOperations)
                results = analyzeGrid(mb)
                print(f"  ... new max block size {results['maxCells']}")

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
