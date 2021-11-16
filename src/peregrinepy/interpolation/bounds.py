# -*- coding: utf-8 -*-

""" bounds.py

Authors:

Kyle Schau

This module contains function related to bounding in an interpolation procedure.

"""

from scipy import spatial
import numpy as np


def ptsInBlkBounds(blk, testPts):

    """
    Takes a block, defines a cube from the extent of that block, and test to see if
    any of the supplied test points are inside that cube.

    Parameters
    ----------

    blk: peregrinepy.block.gridBlock peregrinepy.blocks.grid_block (or a descendant).
         Must have coordinate data populated.

    testPts : np.array
       Numpy array of shape (numPts,3) defining (x,y,z) of each point to test

    Returns
    -------
    bool
       True if any point lies inside the cube, otherwise false.
    """

    xmin = np.min(blk.array["x"])
    xmax = np.max(blk.array["x"])
    ymin = np.min(blk.array["y"])
    ymax = np.max(blk.array["y"])
    zmin = np.min(blk.array["z"])
    zmax = np.max(blk.array["z"])

    # Test if it is possible that any test points lie inside the block
    if (
        xmin > np.max(testPts[:, 0])
        or xmax < np.min(testPts[:, 0])
        or ymin > np.max(testPts[:, 1])
        or ymax < np.min(testPts[:, 1])
        or zmin > np.max(testPts[:, 2])
        or zmax < np.min(testPts[:, 2])
    ):

        return [False] * len(testPts)

    else:
        # Create a list of the corner points of the block that we want to "test" if any test points lie inside
        cube = np.array(
            [
                [
                    blk.array["x"][0, 0, 0],
                    blk.array["y"][0, 0, 0],
                    blk.array["z"][0, 0, 0],
                ],
                [
                    blk.array["x"][-1, 0, 0],
                    blk.array["y"][-1, 0, 0],
                    blk.array["z"][-1, 0, 0],
                ],
                [
                    blk.array["x"][0, -1, 0],
                    blk.array["y"][0, -1, 0],
                    blk.array["z"][0, -1, 0],
                ],
                [
                    blk.array["x"][0, 0, -1],
                    blk.array["y"][0, 0, -1],
                    blk.array["z"][0, 0, -1],
                ],
                [
                    blk.array["x"][-1, -1, 0],
                    blk.array["y"][-1, -1, 0],
                    blk.array["z"][-1, -1, 0],
                ],
                [
                    blk.array["x"][-1, 0, -1],
                    blk.array["y"][-1, 0, -1],
                    blk.array["z"][-1, 0, -1],
                ],
                [
                    blk.array["x"][0, -1, -1],
                    blk.array["y"][0, -1, -1],
                    blk.array["z"][0, -1, -1],
                ],
                [
                    blk.array["x"][-1, -1, -1],
                    blk.array["y"][-1, -1, -1],
                    blk.array["z"][-1, -1, -1],
                ],
            ]
        )

        # In case the block is very curvilinear, we will ad the midpoints of the edges of the block to make
        # the test geometry better match the shape of the block.
        if blk.nx > 4:
            indx = int(blk.nx / 2)
            cube = np.append(
                cube,
                [
                    [
                        blk.array["x"][indx, 0, 0],
                        blk.array["y"][indx, 0, 0],
                        blk.array["z"][indx, 0, 0],
                    ],
                    [
                        blk.array["x"][indx, -1, 0],
                        blk.array["y"][indx, -1, 0],
                        blk.array["z"][indx, -1, 0],
                    ],
                    [
                        blk.array["x"][indx, 0, -1],
                        blk.array["y"][indx, 0, -1],
                        blk.array["z"][indx, 0, -1],
                    ],
                    [
                        blk.array["x"][indx, -1, -1],
                        blk.array["y"][indx, -1, -1],
                        blk.array["z"][indx, -1, -1],
                    ],
                ],
                axis=0,
            )

            # Finally, we will add the centers of each face to the test geometry for a very close representation of the
            # actual block shape
            if blk.ny > 4:
                indx2 = int(blk.ny / 2)
                cube = np.append(
                    cube,
                    [
                        [
                            blk.array["x"][indx, indx2, 0],
                            blk.array["y"][indx, indx2, 0],
                            blk.array["z"][indx, indx2, 0],
                        ],
                        [
                            blk.array["x"][indx, indx2, -1],
                            blk.array["y"][indx, indx2, -1],
                            blk.array["z"][indx, indx2, -1],
                        ],
                    ],
                    axis=0,
                )
            if blk.nz > 4:
                indx2 = int(blk.nz / 2)
                cube = np.append(
                    cube,
                    [
                        [
                            blk.array["x"][indx, 0, indx2],
                            blk.array["y"][indx, 0, indx2],
                            blk.array["z"][indx, 0, indx2],
                        ],
                        [
                            blk.array["x"][indx, -1, indx2],
                            blk.array["y"][indx, -1, indx2],
                            blk.array["z"][indx, -1, indx2],
                        ],
                    ],
                    axis=0,
                )

        if blk.ny > 4:
            indx = int(blk.ny / 2)
            cube = np.append(
                cube,
                [
                    [
                        blk.array["x"][0, indx, 0],
                        blk.array["y"][0, indx, 0],
                        blk.array["z"][0, indx, 0],
                    ],
                    [
                        blk.array["x"][-1, indx, 0],
                        blk.array["y"][-1, indx, 0],
                        blk.array["z"][-1, indx, 0],
                    ],
                    [
                        blk.array["x"][0, indx, -1],
                        blk.array["y"][0, indx, -1],
                        blk.array["z"][0, indx, -1],
                    ],
                    [
                        blk.array["x"][-1, indx, -1],
                        blk.array["y"][-1, indx, -1],
                        blk.array["z"][-1, indx, -1],
                    ],
                ],
                axis=0,
            )
            if blk.nz > 4:
                indx2 = int(blk.nz / 2)
                cube = np.append(
                    cube,
                    [
                        [
                            blk.array["x"][indx, 0, indx2],
                            blk.array["y"][indx, 0, indx2],
                            blk.array["z"][indx, 0, indx2],
                        ],
                        [
                            blk.array["x"][indx, -1, indx2],
                            blk.array["y"][indx, -1, indx2],
                            blk.array["z"][indx, -1, indx2],
                        ],
                    ],
                    axis=0,
                )
        if blk.nz > 4:
            indx = int(blk.nz / 2)
            cube = np.append(
                cube,
                [
                    [
                        blk.array["x"][0, 0, indx],
                        blk.array["y"][0, 0, indx],
                        blk.array["z"][0, 0, indx],
                    ],
                    [
                        blk.array["x"][-1, 0, indx],
                        blk.array["y"][-1, 0, indx],
                        blk.array["z"][-1, 0, indx],
                    ],
                    [
                        blk.array["x"][0, -1, indx],
                        blk.array["y"][0, -1, indx],
                        blk.array["z"][0, -1, indx],
                    ],
                    [
                        blk.array["x"][-1, -1, indx],
                        blk.array["y"][-1, -1, indx],
                        blk.array["z"][-1, -1, indx],
                    ],
                ],
                axis=0,
            )

        hull = spatial.Delaunay(cube)
        hullBool = hull.find_simplex(testPts) >= 0

        return hullBool


def findBounds(mbTo, mbFrom, verboseSearch):
    """
    Compares two multiBlock grids (or descendants) and
    determines which blocks from mbFrom each
    individual block from mbTo reside in, in space.

    Parameters
    ----------

    mbFrom: peregrinepy.multiBlock.grid
       peregrinepy.multiBlock.grid (or a descendant). Must have coordinate data populated

    mbTo: peregrinepy.multiBlock.grid
       peregrinepy.multiBlock.grid (or a descendant). Must have coordinate data populated

    Returns
    -------
    bounding_blocks : list
       List of length len(mbTo) where each entry is itself a list containing the block
       numbers of all the blocks from mbFrom that each block in mbTo reside in, spatially.
    """

    mbFrom.computeMetrics(fdOrder=2)
    mbTo.computeMetrics(fdOrder=2)

    boundingBlocks = []
    for blkTo in mbTo:
        blkTo_xc = np.column_stack(
            (
                blkTo.array["xc"].ravel(),
                blkTo.array["yc"].ravel(),
                blkTo.array["zc"].ravel(),
            )
        )
        ptFound = [False] * len(blkTo_xc)
        currBlkIn = []
        for blkFrom in mbFrom:
            inBlockBool = ptsInBlkBounds(blkFrom, blkTo_xc)
            if True in inBlockBool:
                currBlkIn.append(blkFrom.nblki)
                ptFound = ptFound + inBlockBool
            if not verboseSearch:
                if False not in ptFound:
                    break

        boundingBlocks.append(currBlkIn)

    return boundingBlocks
