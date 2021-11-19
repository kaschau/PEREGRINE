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
            setX = {i for i in corners["x"][index][face.nface - 1]}
            setY = {i for i in corners["y"][index][face.nface - 1]}
            setZ = {i for i in corners["z"][index][face.nface - 1]}
            for testIndex, nblki in enumerate(incompleteBlocks):
                testBlk = mb.getBlock(nblki)
                for testFace in testBlk.faces:
                    if foundFaces[testIndex][testFace.nface - 1]:
                        continue
                    if blk.nblki == testBlk.nblki and face.nface == testFace.nface:
                        continue
                    testSetX = {i for i in corners["x"][testIndex][testFace.nface - 1]}
                    testSetY = {i for i in corners["y"][testIndex][testFace.nface - 1]}
                    testSetZ = {i for i in corners["z"][testIndex][testFace.nface - 1]}
                    if setX == testSetX and setY == testSetY and setZ == testSetZ:
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
        # We will manually find the neighbor later, if needed.
        newSplitFace.neighbor = None
        # If this face is a boundary, then we can add it to the found faces
        if not newSplitFace.bcType.startswith("b"):
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
    pg.grid.create.multiBlockCube(mb, mbDims=[1, 1, 1], dimsPerBlock=[11, 11, 11])

    blk = mb[0]
    face = blk.getFace(3)
    face.neighbor = 0
    face.orientation = "123"
    face.bcType = "b1"
    face = blk.getFace(4)
    face.neighbor = 0
    face.orientation = "123"
    face.bcType = "b1"

    pg.writers.writeGrid(mb, "./unsplit")
    pg.writers.writeConnectivity(mb, "./unsplit")

    incompleteBlocks = []
    foundFaces = []

    cutBlock(mb, 0, "i", 5, incompleteBlocks, foundFaces)
    findInteriorNeighbor(mb, incompleteBlocks, foundFaces)
    findPeriodicNeighbor(mb, incompleteBlocks, foundFaces)
    print(incompleteBlocks, foundFaces)
    assert incompleteBlocks == []
    assert foundFaces == []

    pg.writers.writeGrid(mb, "./")
    pg.writers.writeConnectivity(mb, "./")

    mbRef = pg.multiBlock.grid(2)
    pg.grid.create.multiBlockCube(mbRef, mbDims=[2, 1, 1], dimsPerBlock=[11, 11, 11])
    pg.writers.writeGrid(mbRef, "./reference")
    pg.writers.writeConnectivity(mbRef, "./reference")

    assert verify(mb)
