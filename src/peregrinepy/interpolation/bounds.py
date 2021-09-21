# -*- coding: utf-8 -*-

""" bounds.py

Authors:

Kyle Schau

This module contains function related to bounding in an interpolation procedure.

"""

from scipy import spatial
import numpy as np


def pts_in_blk_bounds(blk, test_pts):

    """
    Takes a block, defines a cube from the extent of that block, and test to see if
    any of the supplied test points are inside that cube.

    Parameters
    ----------

    blk: raptorpy.block.grid_block
       raptorpy.blocks.grid_block (or a descendant). Must have coordinate data populated

    test_pts : np.array
       Numpy array of shape (num_pts,3) defining (x,y,z) of each point to test

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
        xmin > np.max(test_pts[:, 0])
        or xmax < np.min(test_pts[:, 0])
        or ymin > np.max(test_pts[:, 1])
        or ymax < np.min(test_pts[:, 1])
        or zmin > np.max(test_pts[:, 2])
        or zmax < np.min(test_pts[:, 2])
    ):

        return [False] * len(test_pts)

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
        hull_bool = hull.find_simplex(test_pts) >= 0

        return hull_bool


def find_bounds(mb_to, mb_from, verbose_search):
    """
    Compares two multiblock grids (or descendants) and determines which blocks from mb_from each
    individual block from mb_to reside in, in space.

    Parameters
    ----------

    mb_from: raptorpy.multiblock.grid
       raptorpy.multiblock.grid (or a descendant). Must have coordinate data populated

    mb_to: raptorpy.multiblock.grid
       raptorpy.multiblock.grid (or a descendant). Must have coordinate data populated

    Returns
    -------
    bounding_blocks : list
       List of length len(mb_to) where each entry is itself a list containing the block
       numbers of all the blocks from mb_from that each block in mb_to reside in, spatially.
    """

    mb_from.compute_metrics(xc=True, xu=True, xv=True, xw=True)
    mb_to.compute_metrics(xc=True, xu=True, xv=True, xw=True)

    bounding_blocks = []
    for blk_to in mb_to:
        blk_to_xc = np.column_stack(
            (
                blk_to.array["xc"].ravel(),
                blk_to.array["yc"].ravel(),
                blk_to.array["zc"].ravel(),
            )
        )
        pt_found = [False] * len(blk_to_xc)
        curr_blk_in = []
        for blk_from in mb_from:
            in_block_bool = pts_in_blk_bounds(blk_from, blk_to_xc)
            if True in in_block_bool:
                curr_blk_in.append(blk_from.nblki)
                pt_found = pt_found + in_block_bool
            if not verbose_search:
                if False not in pt_found:
                    break

        bounding_blocks.append(curr_blk_in)

    return bounding_blocks
