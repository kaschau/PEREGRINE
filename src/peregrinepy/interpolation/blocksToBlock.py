# -*- coding: utf-8 -*-

""" blocksToBlock.py

Authors:

Kyle Schau

This module contains a function that will interpolate a list of
blocks (blks_from) onto a single block (blk_to)

"""


import numpy as np
from scipy import interpolate


def blocksToBlock(blksFrom, blkTo, function="nearest", smooth=0.5):

    """
    Takes a list of blocks that encompass the blkTo parameter in space and interpolate all
    RAPTOR data from those blocks onto the single block. qv, U,V,W are all interpolated.

    Parameters
    ----------

    blksFrom : list
       List of peregrinepy.blocks.restartBlock

    blkTo : peregrinepy.blocks.restartBlock
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
        Updates attributes of parameter blkTo.
    """
    print(f"Interpolating block {blkTo.nblki}")

    for blk in blksFrom:
        blk.computeMetrics()

    blkTo.computeMetrics()

    # qv interpolation
    blkFromX = np.concatenate(tuple([blk.array["xc"].ravel() for blk in blksFrom]))
    blkFromY = np.concatenate(tuple([blk.array["yc"].ravel() for blk in blksFrom]))
    blkfromZ = np.concatenate(tuple([blk.array["zc"].ravel() for blk in blksFrom]))

    for i in range(blksFrom[0].array["q"].shape[-1]):
        qvFrom = np.concatenate(
            tuple([blk.array["q"][:, :, :, i].ravel() for blk in blksFrom])
        )

        if function == "nearest":
            interpFrom = interpolate.NearestNDInterpolator(
                np.column_stack((blkFromX, blkFromY, blkfromZ)), qvFrom
            )
        else:
            interpFrom = interpolate.Rbf(
                blkFromX,
                blkFromY,
                blkfromZ,
                qvFrom,
                function=function,
                smooth=smooth,
            )

        qvTo = interpFrom(
            blkTo.array["xc"].ravel(),
            blkTo.array["yc"].ravel(),
            blkTo.array["zc"].ravel(),
        )

        # do not allow new extrema to be created
        qvTo = np.where(qvTo > np.max(qvFrom), np.max(qvFrom), qvTo)
        qvTo = np.where(qvTo < np.min(qvFrom), np.min(qvFrom), qvTo)

        blkTo.qv[:, :, :, i] = qvTo.reshape(
            blkTo.qv[:, :, :, i].shape
        )
