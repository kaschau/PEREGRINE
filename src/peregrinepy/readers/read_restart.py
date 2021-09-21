# -*- coding: utf-8 -*-

import numpy as np
import h5py


def read_restart(mb, path="./", nrt=0, animate=True):
    """This function reads in all the HDF5 grid files in :path: and adds the coordinate data to a supplied raptorpy.multiblock.grid object (or one of its descendants)

    Parameters
    ----------

    mb : raptorpy.multiblock.grid (or a descendant)

    file_path : str
        Path to find all the HDF5 grid files to be read in

    Returns
    -------
    None

    """

    for blk in mb:
        variables = ["p", "u", "v", "w", "T"] + blk.species_names[0:-1]

        if animate:
            file_name = f"{path}/q.{nrt:08d}.{blk.nblki:06d}.h5"
        else:
            file_name = f"{path}/q.{blk.nblki:06d}.h5"

        with h5py.File(file_name, "r") as f:

            blk.nrt = list(f["iter"]["nrt"])[0]
            blk.tme = list(f["iter"]["tme"])[0]

            for i, var in enumerate(variables):
                blk.array["q"][1:-1, 1:-1, 1:-1, i] = np.array(
                    f["results"][var]
                ).reshape((blk.ni - 1, blk.nj - 1, blk.nk - 1), order="F")

    # Set the mb values as well
    mb.nrt = mb[0].nrt
    mb.tme = mb[0].tme
