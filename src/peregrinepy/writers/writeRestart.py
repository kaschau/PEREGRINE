import h5py
import numpy as np
from .writeMetaData import restartMetaData


def writeRestart(
    mb,
    path="./",
    gridPath="./",
    animate=True,
    precision="double",
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
        nrt=mb.nrt,
        tme=mb.tme,
    )

    fileName = f"{path}/{metaData.getVarFileName(mb.nrt)}"
    qf = h5py.File(fileName, "w")

    for blk in mb:
        nblki = blk.nblki
        ni, nj, nk = blk.ni, blk.nj, blk.nk


        extentCC = (ni - 1) * (nj - 1) * (nk - 1)

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
                dset[:] = blk.array["Q"][blk.interior + tuple([0])].ravel(order="F")
            except TypeError:
                pass

        names = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
        for j in range(len(names)):
            dsetName = names[j]
            qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf[resS][dsetName]
            dset[:] = blk.array["q"][blk.interior + tuple([j])].ravel(order="F")
        # Compute the nth species here
        dsetName = blk.speciesNames[-1]
        qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
        dset = qf[resS][dsetName]
        if blk.ns > 1:
            dset[:] = 1.0 - np.sum(
                blk.array["q"][blk.interior + tuple([slice(5, None, None)])], axis=-1
            ).ravel(order="F")
        elif blk.ns == 1:
            dset[:] = 1.0

        # Add block to xdmf tree
        blockElem = metaData.addBlockElem(nblki, ni, nj, nk)

        # Add scalar variables to block tree
        names = ["p", "T"] + blk.speciesNames
        if blk.blockType == "solver":
            names.insert(0, "rho")

        for name in names:
            metaData.addScalarToBlockElem(
                blockElem, name, mb.nrt, nblki, ni, nj, nk
            )
        # Add vector variables to block tree
        metaData.addVectorToBlockElem(
            blockElem, "Velocity", ["u", "v", "w"], mb.nrt, nblki, ni, nj, nk
        )

        mb.progress(nblki + 1, f"Writing out restartBlock {nblki}")

    qf.close()

    metaData.saveXdmf(path, mb.nrt)
