# -*- coding: utf-8 -*-

import h5py
import numpy as np
from .writerMetaData import restartMetaData, arbitraryMetaData
from ..mpiComm.mpiUtils import getCommRankSize
from mpi4py.MPI import INT as MPIINT


def registerParallelMetaData(
    mb,
    blocksForProcs,
    gridPath="./",
    precision="double",
    animate=True,
    lump=True,
    arrayName="default",
):
    """This function creates a metaData object when blocks are spread out over multiple
    processors. The strategie is to first create an ordered list of blocks of block
    extents. So we send ni,nj,nk to the zeroth rank, sort that list by block #,
    then broadcast that array.

    Once each processor has the sorted list, we create a meta data object on each processor.
    This writer meta data will only be used by the non zeroth rank if we are writing in
    parallel (via hdf5 + lump). In that case, we use the meta data to write the hdf5 meta data
    on each process. If we are writing in serial, then the non zeroth ranks dont ever use their
    meta data.

    Parameters
    ----------

    mb : peregrinepy.multiBlock.grid (or a descendant)

    blocksForProcs : list
        List of lists with the first index being the rank, second index the block number(s)

    gridPath : str
        Path to grid.

    precision : str ["double","single"]
        Double or single precision writing.

    animate : bool
        Controls the output nameing convention to overwrite previous output or not.

    lump : bool
        Controls whether to output to a single file or one file per block.


    Returns
    -------
    peregrinepy.writer.writerMetaData.restartMetaData

    """

    comm, rank, size = getCommRankSize()
    if arrayName == "default":
        # Add scalar variables to block tree
        names = ["rho", "p", "T"] + mb[0].speciesNames
    else:
        shape = mb[0].array[arrayName].shape
        if len(shape) > 3:
            names = [f"{arrayName}_{i}" for i in range(shape[-1])]
        else:
            names = [arrayName]

    # the mb with Block0 must get a list of all other block's ni,nj,nk
    myNiList = [[blk.ni, blk.nj, blk.nk] for blk in mb]

    blockIndex = 0
    totalNiList = np.zeros((mb.totalBlocks, 3), dtype=np.int32)
    if rank == 0:
        for blk in mb:
            totalNiList[blockIndex, 0] = blk.ni
            totalNiList[blockIndex, 1] = blk.nj
            totalNiList[blockIndex, 2] = blk.nk
            blockIndex += 1

    for sendrank in range(1, size):
        # send ni list
        if rank == sendrank:
            myNiList = np.zeros(3 * mb.nblks, dtype=np.int32)
            mult = 0
            for blk in mb:
                myNiList[3 * mult + 0] = blk.ni
                myNiList[3 * mult + 1] = blk.nj
                myNiList[3 * mult + 2] = blk.nk
                mult += 1
            sendSend = mb.nblks * 3
            comm.Send([myNiList, sendSend, MPIINT], dest=0, tag=rank)
        # recv ni list
        elif rank == 0:
            recvSize = 3 * len(blocksForProcs[sendrank])
            recvBuff = np.zeros(recvSize, dtype=np.int32)
            comm.Recv([recvBuff, recvSize, MPIINT], source=sendrank, tag=sendrank)

            recvBuff = recvBuff.reshape(len(blocksForProcs[sendrank]), 3)
            mult = 0
            for b in recvBuff:
                totalNiList[blockIndex, 0] = b[0]
                totalNiList[blockIndex, 1] = b[1]
                totalNiList[blockIndex, 2] = b[2]
                blockIndex += 1
        else:
            pass

    # Flatten the list, then sort in block order
    if rank == 0:
        totalBlockList = np.array(
            [nblki for b in blocksForProcs for nblki in b], dtype=np.int32
        )
        totalBlockList, totalNiList = (
            np.array(list(t), np.int32)
            for t in zip(*sorted(zip(totalBlockList, totalNiList)))
        )

    # Send the list to all the other processes
    comm.Bcast([totalNiList, mb.totalBlocks * 3, MPIINT], root=0)

    # Create the metaData for all the blocks
    if arrayName == "default":
        metaData = restartMetaData(
            gridPath=gridPath,
            precision=precision,
            animate=animate,
            lump=lump,
            nrt=mb.nrt,
            tme=mb.tme,
        )
    else:
        metaData = arbitraryMetaData(
            arrayName=arrayName,
            gridPath=gridPath,
            precision=precision,
            animate=animate,
            lump=lump,
            nrt=mb.nrt,
            tme=mb.tme,
        )

    for nblki, n in enumerate(totalNiList):
        ni = n[0]
        nj = n[1]
        nk = n[2]
        # Add block to xdmf tree
        blockElem = metaData.addBlockElem(nblki, ni, nj, nk, ng=0)

        for name in names:
            metaData.addScalarToBlockElem(
                blockElem, name, mb.nrt, nblki, ni, nj, nk, ng=0
            )
        if arrayName == "default":
            # Add vector variables to block tree
            metaData.addVectorToBlockElem(
                blockElem, "Velocity", ["u", "v", "w"], mb.nrt, nblki, ni, nj, nk, ng=0
            )

    # Return the meta data
    return metaData


def parallelWriteRestart(
    mb,
    metaData,
    path="./",
):

    comm, rank, size = getCommRankSize()

    if metaData.precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    names = ["rho", "p", "u", "v", "w", "T"] + mb[0].speciesNames

    # If lumping use parallel writer
    if metaData.lump:
        fileName = f"{path}/{metaData.getVarFileName(mb.nrt, None)}"
        qf = h5py.File(fileName, "w", driver="mpio", comm=comm)
        qf.create_group("iter")
        qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
        qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")

        if rank == 0:
            qf["iter"]["nrt"][0] = mb.nrt
            qf["iter"]["tme"][0] = mb.tme
        # If lumping we need to create the meta data for each block
        for nblki in range(mb.totalBlocks):
            qf.create_group(f"results_{nblki:06d}")
            blockMetaData = metaData.tree[0][0][nblki][0]
            ni, nj, nk = (
                int(i) for i in blockMetaData.get("NumberOfElements").split(" ")
            )
            extentCC = (ni - 1) * (nj - 1) * (nk - 1)
            for name in names:
                qf[f"results_{nblki:06d}"].create_dataset(
                    name, shape=(extentCC,), dtype=fdtype
                )

    # Write the hdf5 data
    for blk in mb:
        # update the host views
        blk.updateHostView(["q", "Q"])

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        ng = blk.ng
        nblki = blk.nblki

        # If we arent lumping, each block will open a file, write the iter group
        # then create the datasets with the correct size
        if not metaData.lump:
            fileName = f"{path}/{metaData.getVarFileName(blk.nrt, blk.nblki)}"
            qf = h5py.File(fileName, "w")

            qf.create_group("iter")
            qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
            qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")
            qf["iter"]["nrt"][0] = mb.nrt
            qf["iter"]["tme"][0] = mb.tme
            qf.create_group(f"results_{nblki:06d}")
            names = ["rho", "p", "u", "v", "w", "T"] + blk.speciesNames
            for name in names:
                qf[f"results_{nblki:06d}"].create_dataset(
                    name, shape=(extentCC,), dtype=fdtype
                )

        resS = f"results_{nblki:06d}"
        dsetName = "rho"
        dset = qf[resS][dsetName]
        dset[:] = blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
        for j in range(len(names[1:-1])):
            dsetName = names[j + 1]
            dset = qf[resS][dsetName]
            dset[:] = blk.array["q"][ng:-ng, ng:-ng, ng:-ng, j].ravel(order="F")
        # Compute the nth species here
        dsetName = blk.speciesNames[-1]
        dset = qf[resS][dsetName]
        if blk.ns > 1:
            dset[:] = 1.0 - np.sum(
                blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1
            ).ravel(order="F")
        elif blk.ns == 1:
            dset[:] = 1.0

        if not metaData.lump:
            qf.close()

    if metaData.lump:
        qf.close()

    # Update and write out xdmf
    for grid in metaData.tree[0][0]:
        nblki = int(grid.get("Name")[1::])
        time = grid.find("Time")
        time.set("Value", str(mb.tme))
        for var in grid.findall("Attribute"):
            name = var.get("Name")
            if name != "Velocity":
                text = metaData.getVarFileH5Location(name, mb.nrt, nblki)
                var[0].text = text
            else:
                for v in var.find("DataItem").findall("DataItem"):
                    varName = v.text.split("/")[-1]
                    text = metaData.getVarFileH5Location(varName, mb.nrt, nblki)
                    v.text = text

    if rank == 0:
        metaData.saveXdmf(path, nrt=mb.nrt)


def parallelWriteArbitraryArray(
    mb,
    metaData,
    path="./",
):

    comm, rank, size = getCommRankSize()

    if metaData.precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    arrayName = metaData.arrayName
    shape = mb[0].array[arrayName].shape
    if len(shape) > 3:
        names = [f"{arrayName}_{i}" for i in range(shape[-1])]
    else:
        names = [arrayName]

    # If lumping use parallel writer
    if metaData.lump:
        fileName = f"{path}/{metaData.getVarFileName(mb.nrt, None)}"
        qf = h5py.File(fileName, "w", driver="mpio", comm=comm)
        qf.create_group("iter")
        qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
        qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")

        if rank == 0:
            qf["iter"]["nrt"][0] = mb.nrt
            qf["iter"]["tme"][0] = mb.tme
        # If lumping we need to create the meta data for each block
        for nblki in range(mb.totalBlocks):
            qf.create_group(f"results_{nblki:06d}")
            blockMetaData = metaData.tree[0][0][nblki][0]
            ni, nj, nk = (
                int(i) for i in blockMetaData.get("NumberOfElements").split(" ")
            )
            extentCC = (ni - 1) * (nj - 1) * (nk - 1)
            for name in names:
                qf[f"results_{nblki:06d}"].create_dataset(
                    name, shape=(extentCC,), dtype=fdtype
                )

    # Write the hdf5 data
    for blk in mb:
        # update the host views
        blk.updateHostView([arrayName])

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        ng = blk.ng
        nblki = blk.nblki

        # If we arent lumping, each block will open a file, write the iter group
        # then create the datasets with the correct size
        if not metaData.lump:
            fileName = f"{path}/{metaData.getVarFileName(blk.nrt, blk.nblki)}"
            qf = h5py.File(fileName, "w")

            qf.create_group("iter")
            qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
            qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")
            qf["iter"]["nrt"][0] = mb.nrt
            qf["iter"]["tme"][0] = mb.tme
            qf.create_group(f"results_{nblki:06d}")
            for name in names:
                qf[f"results_{nblki:06d}"].create_dataset(
                    name, shape=(extentCC,), dtype=fdtype
                )

        resS = f"results_{nblki:06d}"
        for j in range(len(names)):
            dsetName = names[j]
            dset = qf[resS][dsetName]
            if len(shape) > 3:
                dset[:] = blk.array[arrayName][ng:-ng, ng:-ng, ng:-ng, j].ravel(
                    order="F"
                )
            else:
                dset[:] = blk.array[arrayName][ng:-ng, ng:-ng, ng:-ng].ravel(order="F")

        if not metaData.lump:
            qf.close()

    if metaData.lump:
        qf.close()

    # Update and write out xdmf
    for grid in metaData.tree[0][0]:
        nblki = int(grid.get("Name")[1::])
        time = grid.find("Time")
        time.set("Value", str(mb.tme))
        for var in grid.findall("Attribute"):
            name = var.get("Name")
            text = metaData.getVarFileH5Location(name, mb.nrt, nblki)
            var[0].text = text

    if rank == 0:
        metaData.saveXdmf(path, nrt=mb.nrt)
