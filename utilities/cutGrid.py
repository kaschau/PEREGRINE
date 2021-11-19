#!/usr/bin/env python3

import numpy as np
import peregrinepy as pg
from verifyGrid import verify


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
    corners = {"x": [], "y": [], "z": []}
    for blk in mb:
        if blk.nblki not in incompleteBlocks:
            continue
        index = incompleteBlocks.index(blk.nblki)
        incompleteFaces = foundFaces[index]

        for var in corners:
            corners[var].append([])

        for found, face in zip(incompleteFaces, blk.faces):
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


def findNeighbor(mb, incompleteBlocks, foundFaces):
    corners = extractCorners(mb, incompleteBlocks, foundFaces)
    assert all(
        [
            len(corners[var]) == len(incompleteBlocks) == len(foundFaces)
            for var in corners
        ]
    )

    for blk in mb:
        # Is this block already complete?
        if blk.nblki not in incompleteBlocks:
            continue
        index = incompleteBlocks.index(blk.nblki)
        # Is this block already complete?
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
            setX = {i for i in corners["x"][index][face.nface - 1]}
            setY = {i for i in corners["y"][index][face.nface - 1]}
            setZ = {i for i in corners["z"][index][face.nface - 1]}
            for testIndex, nblki in enumerate(incompleteBlocks):
                testBlk = mb.getBlock(nblki)
                for testFace in testBlk.faces:
                    if blk.nblki == testBlk.nblki and face.nface == testFace.nface:
                        continue
                    testSetX = {i for i in corners["x"][testIndex][testFace.nface - 1]}
                    testSetY = {i for i in corners["y"][testIndex][testFace.nface - 1]}
                    testSetZ = {i for i in corners["z"][testIndex][testFace.nface - 1]}
                    if setX == testSetX and setY == testSetY and setZ == testSetZ:
                        print("HERE")
                        print(blk.nblki, face.nface)
                        face.neighbor = testBlk.nblki
                        testFace.neighbor = blk.nblki
                        foundFaces[index][face.nface - 1] = True
                        foundFaces[testIndex][testFace.nface - 1] = True


def cutBlock(mb, nblki, cutAxis, cutIndex, incompleteBlocks, foundFaces):

    oldBlk = mb.getBlock(nblki)
    assert (
        getattr(oldBlk, f"n{cutAxis}") > 2
    ), f"Error, trying to cut block {nblki} along axis {cutAxis} with only 2 grid points."
    assert cutIndex < getattr(
        oldBlk, f"n{cutAxis}"
    ), f"Error, trying to cut block {nblki} along axis {cutAxis} at index {cutIndex} > {getattr(oldBlk, f'n{cutAxis}')}."
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
        # We will manually find the neighbor later, if needed.
        newSplitFace.neighbor = None
        # If this face is a boundary, then we can add it to the found faces
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


if __name__ == "__main__":

    mb = pg.multiBlock.grid(1)
    pg.grid.create.multiBlockCube(mb, mbDims=[1, 1, 1], dimsPerBlock=[11, 10, 10])

    pg.writers.writeGrid(mb, "./unsplit")
    pg.writers.writeConnectivity(mb, "./unsplit")

    incompleteBlocks = []
    foundFaces = []

    cutBlock(mb, 0, "i", 5, incompleteBlocks, foundFaces)
    findNeighbor(mb, incompleteBlocks, foundFaces)

    incompleteBlocks = []
    foundFaces = []
    cutBlock(mb, 0, "i", 2, incompleteBlocks, foundFaces)
    print(incompleteBlocks)
    print(foundFaces)
    findNeighbor(mb, incompleteBlocks, foundFaces)

    pg.writers.writeGrid(mb, "./")
    pg.writers.writeConnectivity(mb, "./")

    mbRef = pg.multiBlock.grid(2)
    pg.grid.create.multiBlockCube(mbRef, mbDims=[2, 1, 1], dimsPerBlock=[11, 10, 10])
    pg.writers.writeGrid(mbRef, "./reference")
    pg.writers.writeConnectivity(mbRef, "./reference")

    assert verify(mb)
