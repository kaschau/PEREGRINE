"""
Writing a PEREGRINE grid.

g.h5 carries the grid and everything about it that does not change with the
case: the coordinates of every block, how the blocks connect to each other,
and any partitions the grid has been balanced into.

    g.h5
      totalBlocks                                        attribute
      coordinates_000000/{x,y,z}                         one group per block
      dimensions_000000/{ni,nj,nk}
      connectivity/{neighbor,orientation,bcType,bcFam}    (totalBlocks, 6)
      partitions/16/rank                                  which rank owns each

A grid carries as many partitions side by side as it has been balanced for,
named by the number of ranks, so one grid runs on 4 or on 64 without being
rebalanced. Boundary condition values stay in the case's bcFams.yaml -- they
belong to the case, not to the grid.
"""

import h5py
import numpy as np
from ..misc import progressBar
from .writeMetaData import gridMetaData


def writeGrid(mb, path="./", precision="double", withHalo=False):
    """This function produces an hdf5 file from a peregrinepy.multiBlock.grid (or a descendant) for viewing in Paraview.

    The grid file also carries the connectivity between the blocks, so a grid
    is one file.
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

    Returns
    -------
    None
    """

    if precision == "single":
        fdtype = "float32"
    else:
        fdtype = "float64"

    # Start the xdmf tree
    metaData = gridMetaData(precision)

    gf = h5py.File(f"{path}/g.h5", "w")

    for blk in mb:
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

        nblkiS = f"{blk.nblki:06d}"
        coordS = "coordinates_" + nblkiS
        dimS = "dimensions_" + nblkiS

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
        metaData.addBlockElem(blk.nblki, blk.ni, blk.nj, blk.nk, ng)

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out gridBlock {blk.nblki}")

    gf.attrs["totalBlocks"] = len(mb)
    _writeConnectivity(gf, mb)
    gf.close()

    metaData.saveXdmf(path)


def _writeConnectivity(gf, mb):
    """Store every block's faces as four (totalBlocks, 6) tables."""
    assert sorted(blk.nblki for blk in mb) == list(
        range(len(mb))
    ), "a grid file holds every block of a grid, numbered from zero"

    shape = (len(mb), 6)
    # hdf5 has no None, so a face with no neighbor names -1 and an unset
    # string is an empty one
    neighbor = np.full(shape, -1, dtype=np.int32)
    orientation = np.zeros(shape, dtype=object)
    bcType = np.zeros(shape, dtype=object)
    bcFam = np.zeros(shape, dtype=object)
    for blk in mb:
        for face in blk.faces:
            # a face's cell in the tables; nface counts from one
            mine = blk.nblki, face.nface - 1
            if face.neighbor is not None:
                neighbor[mine] = face.neighbor
            orientation[mine] = face.orientation or ""
            bcType[mine] = face.bcType
            bcFam[mine] = face.bcFam or ""

    if "connectivity" in gf:
        del gf["connectivity"]
    group = gf.create_group("connectivity")
    group.create_dataset("neighbor", data=neighbor)
    for name, table in (
        ("orientation", orientation),
        ("bcType", bcType),
        ("bcFam", bcFam),
    ):
        group.create_dataset(name, data=table, dtype=h5py.string_dtype("utf-8"))


def writePartition(blocksForProcs, path="./"):
    """Add a partition to the grid file at :path:, named by the number of
    ranks it is for. A grid keeps every partition it has been balanced into,
    so one grid runs on any of them. Replaces any partition the grid already
    carries for that many ranks.

    Parameters
    ----------
    blocksForProcs : list
        List of lists, the first index the rank, the second its block number(s)

    path : str
        Path to the directory holding the g.h5 to add the partition to

    Returns
    -------
    None

    """

    size = len(blocksForProcs)
    rank = np.full(sum(len(group) for group in blocksForProcs), -1, dtype=np.int32)
    for r, group in enumerate(blocksForProcs):
        for nblki in group:
            rank[nblki] = r
    assert not (rank == -1).any(), "every block must be owned by a rank"

    with h5py.File(f"{path}/g.h5", "a") as gf:
        if f"partitions/{size}" in gf:
            del gf[f"partitions/{size}"]
        gf.create_dataset(f"partitions/{size}/rank", data=rank)
