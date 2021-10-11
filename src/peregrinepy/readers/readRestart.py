# -*- coding: utf-8 -*-

import numpy as np
import h5py


def readRestart(mb, path="./", nrt=0, animate=True):
    """This function reads in all the HDF5 grid files in :path:
    and adds the coordinate data to a supplied peregrinepy.multiBlock.grid
    object (or one of its descendants)

    Parameters
    ----------

    mb : peregrinepy.multiBlock.grid (or a descendant)

    path : str
        Path to find all the HDF5 grid files to be read in

    Returns
    -------
    None

    """

    for blk in mb:
        variables = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
        if blk.blockType == "solver":
            ng = blk.ng
            readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 0
            readS = np.s_[:, :, :]

        if animate:
            fileName = f"{path}/q.{nrt:08d}.{blk.nblki:06d}.h5"
        else:
            fileName = f"{path}/q.{blk.nblki:06d}.h5"

        with h5py.File(fileName, "r") as f:

            blk.nrt = list(f["iter"]["nrt"])[0]
            blk.tme = list(f["iter"]["tme"])[0]

            for i, var in enumerate(variables):
                blk.array["q"][readS][i] = np.array(
                    f["results"][var]
                ).reshape((blk.ni - 1, blk.nj - 1, blk.nk - 1), order="F")

    # Set the mb values as well
    mb.nrt = mb[0].nrt
    mb.tme = mb[0].tme
