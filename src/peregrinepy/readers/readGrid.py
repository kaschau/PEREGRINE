# -*- coding: utf-8 -*-

import h5py
import numpy as np
from ..misc import progressBar


def readGrid(mb, path="./", justNi=False):
    """
    This function reads in all the HDF5 grid files in
    :path: and adds the coordinate data to a supplied
    peregrinepy.multiBlock.grid object (or one of its descendants)

    Parameters
    ----------
    mb : peregrinepy.multiBlock.grid (or a descendant)

    path : str
        Path to find all the HDF5 grid files to be read in

    justNi: bool
        Whether to just read in block extents or entire grid.


    Returns
    -------
    None

    """
    if justNi:
        assert mb.mbType not in ["restart", "solver"]

    for blk in mb:
        fileName = f"{path}/g.{blk.nblki:06d}.h5"
        if blk.blockType == "solver":
            ng = blk.ng
            readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 0
            readS = np.s_[:, :, :]

        with h5py.File(fileName, "r") as f:
            ni = list(f["dimensions"]["ni"])[0]
            nj = list(f["dimensions"]["nj"])[0]
            nk = list(f["dimensions"]["nk"])[0]

            blk.ni = int(ni)
            blk.nj = int(nj)
            blk.nk = int(nk)

            if not justNi:
                blk.initGridArrays()
                for name in ("x", "y", "z"):
                    blk.array[name][readS] = np.array(f["coordinates"][name]).reshape(
                        (ni, nj, nk), order="F"
                    )

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Reading in gridBlock {blk.nblki}")

        if blk.blockType in ["restart", "solver"]:
            blk.initRestartArrays()
