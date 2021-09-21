# -*- coding: utf-8 -*-

""" blocks_to_block.py

Authors:

Kyle Schau

This module contains a function that will interpolate a list of blocks (blks_from) onto a single block (blk_to)

"""


import numpy as np
from scipy import interpolate


def blocks_to_block(blks_from, blk_to, function="nearest", smooth=0.5):

    """
    Takes a list of blocks that encompass the blk_to parameter in space and interpolate all
    RAPTOR data from those blocks onto the single block. qv, U,V,W are all interpolated.

    Parameters
    ----------

    blks_from : list
       List of raptorpy.blocks.restart_block

    blk_to : raptorpy.blocks.restart_block
       Restart block with populated coordinate data (x,y,z)

    function : string
       String for the desired interpolatin type, options are:
       - nearest  #Uses scipy.interpolate.NearestNDInterpolator
       - Any options to scipy.interpolate.Rbf
          - linear
          - cubic
          - etc...

    smooth : float
       Value of smoothing. Only used for linear, cubic, ... when the scipy.interpolate.Rfb is used.
       Does nothing if nearest function used.

    Returns
    -------
    None
        Updates attributes of parameter blk_to.
    """
    print(f"Interpolating block {blk_to.nblki}")

    for blk in blks_from:
        blk.compute_metrics()

    blk_to.compute_metrics()

    # qv interpolation
    blk_from_x = np.concatenate(tuple([blk.array["xc"].ravel() for blk in blks_from]))
    blk_from_y = np.concatenate(tuple([blk.array["yc"].ravel() for blk in blks_from]))
    blk_from_z = np.concatenate(tuple([blk.array["zc"].ravel() for blk in blks_from]))

    for i in range(5 + blk_to.ns - 1):
        qv_from = np.concatenate(
            tuple([blk.array["q"][1:-1, 1:-1, 1:-1, i].ravel() for blk in blks_from])
        )

        if function == "nearest":
            interp_from = interpolate.NearestNDInterpolator(
                np.column_stack((blk_from_x, blk_from_y, blk_from_z)), qv_from
            )
        else:
            interp_from = interpolate.Rbf(
                blk_from_x,
                blk_from_y,
                blk_from_z,
                qv_from,
                function=function,
                smooth=smooth,
            )

        qv_to = interp_from(
            blk_to.array["xc"].ravel(),
            blk_to.array["yc"].ravel(),
            blk_to.array["zc"].ravel(),
        )

        # do not allow new extrema to be created
        qv_to = np.where(qv_to > np.max(qv_from), np.max(qv_from), qv_to)
        qv_to = np.where(qv_to < np.min(qv_from), np.min(qv_from), qv_to)

        blk_to.qv[1:-1, 1:-1, 1:-1, i] = qv_to.reshape(
            blk_to.qv[1:-1, 1:-1, 1:-1, i].shape
        )

        # Populate halo cells:
        blk_to.array["q"][0, :, :, i] = blk_to.array["q"][i, 1, :, :, i]
        blk_to.array["q"][-1, :, :, i] = blk_to.array["q"][i, -2, :, :, i]
        blk_to.array["q"][:, 0, :, i] = blk_to.array["q"][i, :, 1, :, i]
        blk_to.array["q"][:, -1, :, i] = blk_to.array["q"][i, :, -2, :, i]
        blk_to.array["q"][:, :, 0, i] = blk_to.array["q"][i, :, :, 1, i]
        blk_to.array["q"][:, :, -1, i] = blk_to.array["q"][i, :, :, -2, i]
