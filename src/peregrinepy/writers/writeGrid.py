# -*- coding: utf-8 -*-
import h5py
import numpy as np
from ..misc import progressBar
from .writerMetaData import gridMetaData


def writeGrid(mb, path="./", precision="double", withHalo=False, lump=False):
    """This function produces an hdf5 file from a peregrinepy.multiBlock.grid (or a descendant) for viewing in Paraview.
    Parameters
    ----------
    mb : peregrinepy.multiBlock.grid (or a descendant)

    file_path : str
        Path to location to write output files

    precision : str
        Options - 'single' for single precision
                  'double' for double precision

    withHalo : bool
        Whether we write out with halo

    lump : bool
        Whether to write out a lumped file

    Returns
    -------
    None
    """

    if precision == "single":
        fdtype = "float32"
    else:
        fdtype = "float64"

    # Start the xdmf tree
    metaData = gridMetaData(precision, lump)

    # If we are lumping the files, open it here
    if lump:
        gf = h5py.File(f"{path}/grid.h5", "w")

    for blk in mb:
        if withHalo and blk.blockType == "solver":
            ng = blk.ng
        else:
            ng = 0

        if blk.blockType == "solver":
            if withHalo:
                writeS = np.s_[:, :, :]
            else:
                writeS = np.s_[blk.ng : -blk.ng, blk.ng : -blk.ng, blk.ng : -blk.ng]
        else:
            writeS = np.s_[:, :, :]

        nblkiS = f"{blk.nblki:06d}"
        coordS = "coordinates_" + nblkiS
        dimS = "dimensions_" + nblkiS

        # If we are doing serial output, open the file here
        if not lump:
            gf = h5py.File(f"{path}/g.{nblkiS}.h5", "w")

        gf.create_group(coordS)
        gf.create_group(dimS)

        gf[dimS].create_dataset("ni", shape=(1,), dtype="int32")
        gf[dimS].create_dataset("nj", shape=(1,), dtype="int32")
        gf[dimS].create_dataset("nk", shape=(1,), dtype="int32")

        dset = gf[dimS]["ni"]
        dset[0] = blk.ni + 2 * ng
        dset = gf[dimS]["nj"]
        dset[0] = blk.nj + 2 * ng
        dset = gf[dimS]["nk"]
        dset[0] = blk.nk + 2 * ng

        extent = (blk.ni + 2 * ng) * (blk.nj + 2 * ng) * (blk.nk + 2 * ng)
        gf[coordS].create_dataset("x", shape=(extent,), dtype=fdtype)
        gf[coordS].create_dataset("y", shape=(extent,), dtype=fdtype)
        gf[coordS].create_dataset("z", shape=(extent,), dtype=fdtype)

        dset = gf[coordS]["x"]
        dset[:] = blk.array["x"][writeS].ravel(order="F")
        dset = gf[coordS]["y"]
        dset[:] = blk.array["y"][writeS].ravel(order="F")
        dset = gf[coordS]["z"]
        dset[:] = blk.array["z"][writeS].ravel(order="F")

        # Add block to xdmf tree
        metaData.addBlockElem(
            blk.nblki, blk.ni + 2 * ng, blk.nj + 2 * ng, blk.nk + 2 * ng, ng
        )

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out gridBlock {blk.nblki}")
        if not lump:
            gf.close()

    if lump:
        gf.close()

    metaData.saveXdmf(path)
