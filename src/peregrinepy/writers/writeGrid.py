# -*- coding: utf-8 -*-
import h5py
import numpy as np
from ..misc import progressBar
from .xdmfTemplates import gridXdmf


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
    xdmfTree = gridXdmf(path, precision, lump)

    # If we are lumping the files, open it here
    if lump:
        f = h5py.File(f"{path}/grid.h5", "w")

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
            f = h5py.File(f"{path}/g.{nblkiS}.h5", "w")

        f.create_group(coordS)
        f.create_group(dimS)

        f[dimS].create_dataset("ni", shape=(1,), dtype="int32")
        f[dimS].create_dataset("nj", shape=(1,), dtype="int32")
        f[dimS].create_dataset("nk", shape=(1,), dtype="int32")

        dset = f[dimS]["ni"]
        dset[0] = blk.ni + 2 * ng
        dset = f[dimS]["nj"]
        dset[0] = blk.nj + 2 * ng
        dset = f[dimS]["nk"]
        dset[0] = blk.nk + 2 * ng

        extent = (blk.ni + 2 * ng) * (blk.nj + 2 * ng) * (blk.nk + 2 * ng)
        f[coordS].create_dataset("x", shape=(extent,), dtype=fdtype)
        f[coordS].create_dataset("y", shape=(extent,), dtype=fdtype)
        f[coordS].create_dataset("z", shape=(extent,), dtype=fdtype)

        dset = f[coordS]["x"]
        dset[:] = blk.array["x"][writeS].ravel(order="F")
        dset = f[coordS]["y"]
        dset[:] = blk.array["y"][writeS].ravel(order="F")
        dset = f[coordS]["z"]
        dset[:] = blk.array["z"][writeS].ravel(order="F")

        # Add block to xdmf tree
        xdmfTree.addBlockElem(blk.nblki, blk.ni, blk.nj, blk.nk, ng)

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out gridBlock {blk.nblki}")
        if not lump:
            f.close()

    if lump:
        f.close()

    xdmfTree.saveXdmf()
