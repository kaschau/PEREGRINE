# -*- coding: utf-8 -*-

import h5py
import numpy as np


def readGrid(mb, path="./"):
    """
    This function reads in all the HDF5 grid files in
    :path: and adds the coordinate data to a supplied
    peregrinepy.multiBlock.grid object (or one of its descendants)

    Parameters
    ----------

    Returns
    -------
    None

    """

    for blk in mb:
        fileName = f"{path}/gv.{blk.nblki:06d}.h5"

        with h5py.File(fileName, "r") as f:
            ni = list(f["dimensions"]["ni"])[0]
            nj = list(f["dimensions"]["nj"])[0]
            nk = list(f["dimensions"]["nk"])[0]

            blk.ni = ni
            blk.nj = nj
            blk.nk = nk

            blk.initGridArrays()

            for name in ("x", "y", "z"):
                if blk.blockType == "solver":
                    blk.array[name][1:-1, 1:-1, 1:-1] = np.array(
                        f["coordinates"][name]
                    ).reshape((ni, nj, nk), order="F")
                else:
                    blk.array[name][:] = np.array(f["coordinates"][name]).reshape(
                        (ni, nj, nk), order="F"
                    )

        if blk.blockType in ["restartBlock", "solverBlock"]:
            blk.initRestartArrays()
        if blk.blockType == "solverBlock":
            blk.initSolverArrays()
