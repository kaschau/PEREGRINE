import h5py
from .writeMetaData import arbitraryMetaData


def writeArbitraryArray(
    mb,
    arrayName,
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
    assert mb.mbType == "solver"

    if precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    # Start the xdmf tree
    metaData = arbitraryMetaData(
        arrayName=arrayName,
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

        array = blk.array[arrayName]
        arrayDim = len(array.shape)
        if arrayDim > 3:
            names = [f"{arrayName}_{i}" for i in range(array.shape[-1])]
        else:
            names = [arrayName]

        for j in range(len(names)):
            dsetName = names[j]
            qf[resS].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf[resS][dsetName]
            if arrayDim > 3:
                dset[:] = array[blk.interior + tuple([j])].ravel(order="F")
            else:
                dset[:] = array[blk.interior].ravel(order="F")

        # Add block to xdmf tree
        blockElem = metaData.addBlockElem(nblki, ni, nj, nk)

        for name in names:
            metaData.addScalarToBlockElem(
                blockElem, name, mb.nrt, nblki, ni, nj, nk
            )

    qf.close()

    metaData.saveXdmf(path, mb.nrt)
