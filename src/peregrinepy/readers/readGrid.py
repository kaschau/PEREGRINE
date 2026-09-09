"""
Reading a PEREGRINE grid.

One g.h5 holds the coordinates of every block, the connectivity between them,
and any partitions the grid has been balanced into. See
peregrinepy/writers/writeGrid.py for the layout.
"""

import h5py
import numpy as np
from ..misc import progressBar


def readGrid(mb, path="./", justNi=False):
    """
    This function reads in all the HDF5 grid files in
    :path: and adds the coordinate data to a supplied
    peregrinepy.multiBlock.grid object (or one of its descendants)

    Parameters
    ----------
    mb : peregrinepy.multiBlock.grid (or a descendant)

    path : str
        Path to find all the HDF5 grid files to be read in

    justNi: bool
        Whether to just read in block extents or entire grid.


    Returns
    -------
    None

    """
    if justNi:
        assert mb.mbType not in ["restart", "solver"]

    gf = h5py.File(f"{path}/g.h5", "r")

    for blk in mb:
        if blk.blockType == "solver":
            ng = blk.ng
            readS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 0
            readS = np.s_[:, :, :]

        nblkiS = f"{blk.nblki:06d}"
        coordS = "coordinates_" + nblkiS
        dimS = "dimensions_" + nblkiS

        ni = list(gf[dimS]["ni"])[0]
        nj = list(gf[dimS]["nj"])[0]
        nk = list(gf[dimS]["nk"])[0]

        blk.ni = int(ni)
        blk.nj = int(nj)
        blk.nk = int(nk)

        if not justNi:
            blk.initGridArrays()
            for name in ("x", "y", "z"):
                blk.array[name][readS] = np.array(gf[coordS][name]).reshape(
                    (ni, nj, nk), order="F"
                )

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Reading in gridBlock {blk.nblki}")

    gf.close()


def readConnectivity(mb, gridPath):
    """
    This function reads the connectivity carried by the grid file at
    :gridPath: and adds it to the supplied peregrinepy.multiBlock object

    Parameters
    ----------
    mb : peregrine.multiBlock.topology (or a descendant)

    gridPath : str
        Path to the directory holding the g.h5 to be read in

    Returns
    -------
    None
        Adds the connectivity information to mb

    """

    with h5py.File(f"{gridPath}/g.h5", "r") as gf:
        group = gf["connectivity"]
        neighbor = np.array(group["neighbor"])
        tables = {
            name: group[name].asstr()[:] for name in ("orientation", "bcType", "bcFam")
        }
        mb.totalBlocks = int(gf.attrs["totalBlocks"])

    for blk in mb:
        for face in blk.faces:
            nface = face.nface - 1
            # the face setters take python types, not numpy ones, and read an
            # empty string as unset and -1 as no neighbor
            face.bcType = str(tables["bcType"][blk.nblki, nface])
            face.bcFam = str(tables["bcFam"][blk.nblki, nface]) or None
            face.orientation = str(tables["orientation"][blk.nblki, nface]) or None
            n = int(neighbor[blk.nblki, nface])
            face.neighbor = None if n == -1 else n


def readTotalBlocks(gridPath="./Grid"):
    """How many blocks the grid at :gridPath: holds.

    Parameters
    ----------
    gridPath : str
        Path to the directory holding the g.h5 to be read in

    Returns
    -------
    int

    """

    with h5py.File(f"{gridPath}/g.h5", "r") as gf:
        return int(gf.attrs["totalBlocks"])


def readPartition(gridPath="./Grid", nProcs=1):
    """The blocks each of nProcs ranks owns, or None if the grid carries no
    partition for that many.

    Parameters
    ----------
    gridPath : str
        Path to the directory holding the g.h5 to be read in

    nProcs : int
        How many ranks the partition is for

    Returns
    -------
    list or None
        List of lists, the first index the rank, the second its block number(s)

    """

    with h5py.File(f"{gridPath}/g.h5", "r") as gf:
        if f"partitions/{nProcs}" not in gf:
            return None
        rank = np.array(gf[f"partitions/{nProcs}/rank"])
        totalBlocks = int(gf.attrs["totalBlocks"])

    assert rank.size == totalBlocks, (
        f"the {nProcs} rank partition covers {rank.size} blocks, "
        f"but this grid has {totalBlocks}"
    )
    return [[int(n) for n in np.flatnonzero(rank == r)] for r in range(nProcs)]


def listPartitions(gridPath="./Grid"):
    """The rank counts the grid at :gridPath: has been partitioned for.

    Parameters
    ----------
    gridPath : str
        Path to the directory holding the g.h5 to be read in

    Returns
    -------
    list of int

    """

    with h5py.File(f"{gridPath}/g.h5", "r") as gf:
        return sorted(int(n) for n in gf["partitions"]) if "partitions" in gf else []
