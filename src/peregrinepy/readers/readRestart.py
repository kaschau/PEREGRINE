import numpy as np
import h5py
from ..misc import progressBar


def readRestart(mb, path="./", nrt=0, animate=True, lump=True):
    """This function reads in all the HDF5 grid files in :path:
    and adds the coordinate data to a supplied peregrinepy.multiBlock.restart
    object (or one of its descendants)

    Parameters
    ----------

    mb : peregrinepy.multiBlock.restart (or a descendant)

    path : str
        Path to find all the HDF5 grid files to be read in

    animate : bool
        Whether we are appending nrt to the file name.

    lump : bool
        Whether we are reading a lumped file or not.

    Returns
    -------
    None

    """

    # If we are lumping, open the file here
    if lump:
        if animate:
            fileName = f"{path}/q.{nrt:08d}.h5"
        else:
            fileName = f"{path}/q.h5"
        qf = h5py.File(fileName, "r")

    for blk in mb:
        # Create the "q" array
        blk.initRestartArrays()

        variables = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
        if blk.blockType == "solver":
            ng = blk.ng
            readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 0
            readS = np.s_[:, :, :]

        # If we are NOT lumping, open the file here
        if not lump:
            if animate:
                fileName = f"{path}/q.{nrt:08d}.{blk.nblki:06d}.h5"
            else:
                fileName = f"{path}/q.{blk.nblki:06d}.h5"

            qf = h5py.File(fileName, "r")

        blk.nrt = int(list(qf["iter"]["nrt"])[0])
        blk.tme = float(list(qf["iter"]["tme"])[0])

        for i, var in enumerate(variables):
            try:
                blk.array["q"][readS + tuple([i])] = np.array(
                    qf[f"results_{blk.nblki:06d}"][var]
                ).reshape((blk.ni - 1, blk.nj - 1, blk.nk - 1), order="F")
            except KeyError:
                if blk.nblki == 0:
                    print(f"Warning, {var} not found in restart. Leaving as is.")

        if mb.mbType == "restart":
            progressBar(blk.nblki + 1, len(mb), f"Reading in restartBlock {blk.nblki}")
        if not lump:
            qf.close()

        # If we are a solver, just fill in the halos with the nearest value to
        # help with boundary conditions and eos calculations from dividing by 0
        # and so on.
        if mb.mbType == "solver":
            blk.array["q"][0:ng, :, :, :] = blk.array["q"][[ng], :, :, :]
            blk.array["q"][-ng::, :, :, :] = blk.array["q"][[-ng - 1], :, :, :]
            blk.array["q"][:, 0:ng, :, :] = blk.array["q"][:, [ng], :, :]
            blk.array["q"][:, -ng::, :, :] = blk.array["q"][:, [-ng - 1], :, :]
            blk.array["q"][:, :, 0:ng, :] = blk.array["q"][:, :, [ng], :]
            blk.array["q"][:, :, -ng::, :] = blk.array["q"][:, :, [-ng - 1], :]

    if lump:
        qf.close()

    # Set the mb values as well
    mb.nrt = mb[0].nrt
    mb.tme = mb[0].tme
