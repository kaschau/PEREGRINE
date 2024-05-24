import h5py
import numpy as np
from ..misc import progressBar
from .writeMetaData import restartMetaData


def writeRestart(
    mb,
    path="./",
    gridPath="./",
    animate=True,
    precision="double",
    withHalo=False,
    lump=True,
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
    metaData = restartMetaData(
        gridPath=gridPath,
        precision=precision,
        animate=animate,
        lump=lump,
        nrt=mb.nrt,
        tme=mb.tme,
    )

    # If writing a lumped file, open it here
    if lump:
        fileName = f"{path}/{metaData.getVarFileName(mb.nrt, None)}"
        qf = h5py.File(fileName, "w")

    for blk in mb:
        nblki = blk.nblki
        ni, nj, nk = blk.ni, blk.nj, blk.nk

        if blk.blockType == "solver":
            if withHalo:
                writeS = np.s_[:, :, :]
                ng = blk.ng
            else:
                writeS = np.s_[blk.ng : -blk.ng, blk.ng : -blk.ng, blk.ng : -blk.ng]
                ng = 0
        else:
            writeS = np.s_[:, :, :]
            ng = 0

        extentCC = (ni + 2 * ng - 1) * (nj + 2 * ng - 1) * (nk + 2 * ng - 1)

        # If not lumping the file, open it here
        if not lump:
            fileName = f"{path}/{metaData.getVarFileName(blk.nrt, blk.nblki)}"
            qf = h5py.File(fileName, "w")

        if "iter" not in qf.keys():
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
                pass

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
        blockElem = metaData.addBlockElem(nblki, ni, nj, nk, ng)

        # Add scalar variables to block tree
        names = ["p", "T"] + blk.speciesNames
        if blk.blockType == "solver":
            names.insert(0, "rho")

        for name in names:
            metaData.addScalarToBlockElem(
                blockElem, name, mb.nrt, nblki, ni, nj, nk, ng
            )
        # Add vector variables to block tree
        metaData.addVectorToBlockElem(
            blockElem, "Velocity", ["u", "v", "w"], mb.nrt, nblki, ni, nj, nk, ng
        )

        if mb.mbType == "restart":
            progressBar(nblki + 1, len(mb), f"Writing out restartBlock {nblki}")

        if not lump:
            qf.close()

    if lump:
        qf.close()

    metaData.saveXdmf(path, mb.nrt)
