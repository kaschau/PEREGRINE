from scipy import spatial
import numpy as np
from ..misc import progressBar


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
    if blk.ni > 4:
        indx = int(blk.ni / 2)
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
        if blk.nj > 4:
            indx2 = int(blk.nj / 2)
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
        if blk.nk > 4:
            indx2 = int(blk.nk / 2)
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

    if blk.nj > 4:
        indx = int(blk.nj / 2)
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
        if blk.nk > 4:
            indx2 = int(blk.nk / 2)
            cube = np.append(
                cube,
                [
                    [
                        blk.array["x"][0, indx, indx2],
                        blk.array["y"][0, indx, indx2],
                        blk.array["z"][0, indx, indx2],
                    ],
                    [
                        blk.array["x"][-1, indx, indx2],
                        blk.array["y"][-1, indx, indx2],
                        blk.array["z"][-1, indx, indx2],
                    ],
                ],
                axis=0,
            )
    if blk.nk > 4:
        indx = int(blk.nk / 2)
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

    mbFrom.computeMetrics(xcOnly=True)
    mbTo.computeMetrics(xcOnly=True)

    boundingBlocks = []
    nblks = mbTo.nblks

    fromBounds = np.zeros((mbFrom.nblks, 6))
    for blkFrom in mbFrom:
        fromBounds[blkFrom.nblki, 0] = np.min(blkFrom.array["x"])
        fromBounds[blkFrom.nblki, 1] = np.max(blkFrom.array["x"])
        fromBounds[blkFrom.nblki, 2] = np.min(blkFrom.array["y"])
        fromBounds[blkFrom.nblki, 3] = np.max(blkFrom.array["y"])
        fromBounds[blkFrom.nblki, 4] = np.min(blkFrom.array["z"])
        fromBounds[blkFrom.nblki, 5] = np.max(blkFrom.array["z"])

    for blkTo in mbTo:
        blkTo_xc = np.column_stack(
            (
                blkTo.array["xc"].ravel(),
                blkTo.array["yc"].ravel(),
                blkTo.array["zc"].ravel(),
            )
        )
        toBounds = np.zeros(6)
        toBounds[0] = np.min(blkTo_xc[:, 0])
        toBounds[1] = np.max(blkTo_xc[:, 0])
        toBounds[2] = np.min(blkTo_xc[:, 1])
        toBounds[3] = np.max(blkTo_xc[:, 1])
        toBounds[4] = np.min(blkTo_xc[:, 2])
        toBounds[5] = np.max(blkTo_xc[:, 2])
        ptFound = np.zeros(len(blkTo_xc), dtype=bool)
        currBlkIn = []
        for blkFrom in mbFrom:
            # Test if it is possible that any test points lie inside the block
            if (
                fromBounds[blkFrom.nblki, 0] > toBounds[1]
                or fromBounds[blkFrom.nblki, 1] < toBounds[0]
                or fromBounds[blkFrom.nblki, 2] > toBounds[3]
                or fromBounds[blkFrom.nblki, 3] < toBounds[2]
                or fromBounds[blkFrom.nblki, 4] > toBounds[5]
                or fromBounds[blkFrom.nblki, 5] < toBounds[4]
                and not verboseSearch
            ):
                pass
            else:
                inBlockBool = ptsInBlkBounds(blkFrom, blkTo_xc)
                if True in inBlockBool:
                    currBlkIn.append(blkFrom.nblki)
                    ptFound += inBlockBool
                if not verboseSearch:
                    if False not in ptFound:
                        break

        boundingBlocks.append(currBlkIn)
        progressBar(blkTo.nblki + 1, nblks, f"Finding block {blkTo.nblki} bounds")

    return boundingBlocks
