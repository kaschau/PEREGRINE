# -*- coding: utf-8 -*-

import h5py
import numpy as np
from ..misc import progressBar


def readGrid(mb, path="./", lump=False, justNi=False):
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

    # If were reading a lumped open the file here
    if lump:
        f = h5py.File(f"{path}/grid.h5", "r")

    for blk in mb:
        if blk.blockType == "solver":
            ng = blk.ng
            readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 0
            readS = np.s_[:, :, :]

        # If were not reading a lumped file, open it here
        if not lump:
            fileName = f"{path}/g.{blk.nblki:06d}.h5"
            f = h5py.File(fileName, "r")

        nblkiS = f"{blk.nblki:06d}"
        coordS = "coordinates_" + nblkiS
        dimS = "dimensions_" + nblkiS

        ni = list(f[dimS]["ni"])[0]
        nj = list(f[dimS]["nj"])[0]
        nk = list(f[dimS]["nk"])[0]

        blk.ni = int(ni)
        blk.nj = int(nj)
        blk.nk = int(nk)

        if not justNi:
            blk.initGridArrays()
            for name in ("x", "y", "z"):
                blk.array[name][readS] = np.array(f[coordS][name]).reshape(
                    (ni, nj, nk), order="F"
                )

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Reading in gridBlock {blk.nblki}")

        if blk.blockType in ["restart", "solver"]:
            blk.initRestartArrays()
