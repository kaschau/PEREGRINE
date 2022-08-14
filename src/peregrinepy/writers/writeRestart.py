import h5py
import numpy as np
from ..misc import progressBar
from .xdmfTemplates import restartXdmf


def writeRestart(
    mb, path="./", gridPath="./", animate=True, precision="double", lump=False
):
    """This function produces an hdf5 file from a peregrinepy.multiBlock.restart for viewing in Paraview.

    Parameters
    ----------

    mb : peregrinepy.multiBlock.restart

    filePath : str
        Path to location to write output files

    precision : str
        Options - 'single' for single precision
                  'double' for double precision

    Returns
    -------
    None

    """

    if precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    # Start the xdmf tree
    xdmfTree = restartXdmf(path=path, precision=precision, animate=animate, lump=lump)

    # If writing a lumped file, open it here
    if lump:
        if animate:
            fileName = f"{path}/q.{mb.nrt:08d}.h5"
        else:
            fileName = f"{path}/q.h5"
        qf = h5py.File(fileName, "w")

    for blk in mb:
        nblki = blk.nblki
        ni, nj, nk = blk.ni, blk.nj, blk.nk

        extentCC = (ni - 1) * (nj - 1) * (nk - 1)

        if blk.blockType == "solver":
            ng = blk.ng
            writeS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 1
            writeS = np.s_[:, :, :]

        if not lump:
            if animate:
                fileName = f"{path}/q.{mb.nrt:08d}.{nblki:06d}.h5"
            else:
                fileName = f"{path}/q.{nblki:06d}.h5"

            # If not lumping the file, open it here
            qf = h5py.File(fileName, "w")

        if not lump or nblki == mb[0].nblki:
            qf.create_group("iter")
            qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
            qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")

            dset = qf["iter"]["nrt"]
            dset[0] = blk.nrt
            dset = qf["iter"]["tme"]
            dset[0] = blk.tme

        resS = f"results_{nblki:06d}"
        qf.create_group(resS)

        if blk.blockType == "solver":
            dsetName = "rho"
            qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf[resS][dsetName]
            try:
                dset[:] = blk.array["Q"][writeS + tuple([0])].ravel(order="F")
            except TypeError:
                # Sometime we may not have density, so just make a zero array
                dset[:] = np.zeros(blk.array["q"][:, :, :, 0][writeS].shape).ravel(
                    order="F"
                )
        names = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
        for j in range(len(names)):
            dsetName = names[j]
            qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf[resS][dsetName]
            dset[:] = blk.array["q"][writeS + tuple([j])].ravel(order="F")
        # Compute the nth species here
        dsetName = blk.speciesNames[-1]
        qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
        dset = qf[resS][dsetName]
        if blk.ns > 1:
            dset[:] = 1.0 - np.sum(
                blk.array["q"][writeS + tuple([slice(5, None, None)])], axis=-1
            ).ravel(order="F")
        elif blk.ns == 1:
            dset[:] = 1.0

        # Add block to xdmf tree
        blockElem = xdmfTree.addBlockElem(nblki, ni, nj, nk, ng, mb.tme)

        # Add scalar variables to block tree
        names = ["p", "T"] + blk.speciesNames
        if blk.blockType == "solver":
            names.insert(0, "rho")

        for name in names:
            xdmfTree.addScalarToBlockElem(blockElem, name, nblki, mb.nrt, ni, nj, nk)
        # Add vector variables to block tree
        xdmfTree.addVectorToBlockElem(
            blockElem, "Velocity", ["u", "v", "w"], nblki, mb.nrt, ni, nj, nk
        )

        if mb.mbType == "restart":
            progressBar(nblki + 1, len(mb), f"Writing out restartBlock {nblki}")

        if not lump:
            qf.close()

    if lump:
        qf.close()
    xdmfTree.saveXdmf()
