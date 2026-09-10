"""
Writing a PEREGRINE grid.

g.h5 carries the grid and everything about it that does not change with the
case: the coordinates of every block, how the blocks connect to each other,
and any partitions the grid has been balanced into.

    g.h5
      totalBlocks                                        attribute
      coordinates_000000/{x,y,z}                         one group per block

Coordinates are stored (nk, nj, ni) so a piece of a block is a contiguous
hyperslab, and a block's extents are the shape of its datasets rather than a
second thing that can disagree with them.
      connectivity/{neighbor,orientation,bcType,bcFam}    (totalBlocks, 6)
      partitions/1x1/rank                                 the base grid
      partitions/64x4/rank                                which rank owns each

A grid carries as many partitions side by side as it has been balanced for,
named ranks x ranksPerNode, so one grid runs on 64 ranks of 4 per node or of
8 without being rebalanced -- the two place blocks differently, because what
crosses a node costs more than what stays on one. Partition 1x1 is the base
grid, every block on one rank, and is written with the grid so it is never a
special case. Boundary condition values stay in the case's bcFams.yaml -- they
belong to the case, not to the grid.
"""

import h5py
import numpy as np
from ..decomposition import cutTable
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

        coordS = gf.create_group(f"coordinates_{blk.nblki:06d}")
        for name in ("x", "y", "z"):
            coordS.create_dataset(
                name,
                data=np.ascontiguousarray(blk.array[name][writeS].T),
                dtype=fdtype,
            )

        # Add block to xdmf tree
        metaData.addBlockElem(blk.nblki, blk.ni, blk.nj, blk.nk, ng)

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out gridBlock {blk.nblki}")

    gf.attrs["totalBlocks"] = len(mb)
    _writeConnectivity(gf, mb)
    # the base grid is a partition like any other: one rank owning all of it
    gf.create_dataset("partitions/1x1/rank", data=np.zeros(len(mb), dtype=np.int32))
    gf.close()

    metaData.saveXdmf(path)


def _writeConnectivity(gf, mb):
    """Store every block's faces as four (totalBlocks, 6) tables."""
    assert sorted(blk.nblki for blk in mb) == list(
        range(len(mb))
    ), "a grid file holds every block of a grid, numbered from zero"

    shape = (len(mb), 6)
    # hdf5 has no None: -1 is no neighbor, an empty string is unset
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


def writePartition(mb, blocksForProcs, ranksPerNode, path="./"):
    """Add a partition to the grid file at :path:, named by the number of
    ranks it is for. A grid keeps every partition it has been balanced into,
    so one grid runs on any of them. Replaces any partition the grid already
    carries for that many ranks.

    A partition whose blocks are pieces of the base grid also stores the cut
    table that says which slab of which base block each piece is, and the
    pieces' own connectivity. One whose blocks are the base blocks needs
    neither, and uses the grid's.

    Parameters
    ----------
    mb : peregrinepy.multiBlock.grid (or a descendant)
        The blocks being partitioned, base blocks or pieces of them

    blocksForProcs : list
        List of lists, the first index the rank, the second its block number(s)

    ranksPerNode : int
        How many of those ranks share a node, which is what the placement was
        optimized for

    path : str
        Path to the directory holding the g.h5 to add the partition to

    Returns
    -------
    None

    """

    name = f"{len(blocksForProcs)}x{ranksPerNode}"
    rank = np.full(sum(len(group) for group in blocksForProcs), -1, dtype=np.int32)
    for r, group in enumerate(blocksForProcs):
        for nblki in group:
            rank[nblki] = r
    assert not (rank == -1).any(), "every block must be owned by a rank"

    with h5py.File(f"{path}/g.h5", "a") as gf:
        if f"partitions/{name}" in gf:
            del gf[f"partitions/{name}"]
        group = gf.create_group(f"partitions/{name}")
        group.create_dataset("rank", data=rank)

        if any(blk.baseSlice is not None for blk in mb):
            group.create_dataset("cuts", data=cutTable(mb))
            _writeConnectivity(group, mb)
