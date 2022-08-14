# -*- coding: utf-8 -*-

import h5py
import numpy as np
from .writerMetaData import restartMetaData
from ..mpiComm.mpiUtils import getCommRankSize
from mpi4py.MPI import INT as MPIINT


def registerParallelMetaData(
    mb, blocksForProcs, gridPath="./", precision="double", animate=True, lump=False
):

    comm, rank, size = getCommRankSize()
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
            for blk in recvBuff:
                totalNiList[blockIndex, 0] = blk[0]
                totalNiList[blockIndex, 1] = blk[1]
                totalNiList[blockIndex, 2] = blk[2]
                blockIndex += 1
        else:
            pass

    # Flatten the list, then sort in block order
    totalBlockList = np.array(
        [nblki for b in blocksForProcs for nblki in b], dtype=np.int32
    )
    if rank == 0:
        totalBlockList, totalNiList = (
            np.array(list(t), np.int32)
            for t in zip(*sorted(zip(totalBlockList, totalNiList)))
        )

    # Send the list to all the other processes
    comm.Bcast([totalBlockList, mb.totalBlocks, MPIINT], root=0)

    # Create the metaData for all the blocks
    metaData = restartMetaData(
        gridPath=gridPath,
        precision=precision,
        animate=animate,
        lump=lump,
        nrt=mb.nrt,
        tme=mb.tme,
    )

    for nblki, n in zip(totalBlockList, totalNiList):
        ni = n[0]
        nj = n[1]
        nk = n[2]
        # Add block to xdmf tree
        blockElem = metaData.addBlockElem(nblki, ni, nj, nk, ng=0)

        # Add scalar variables to block tree
        names = ["rho", "p", "T"] + blk.speciesNames

        for name in names:
            metaData.addScalarToBlockElem(blockElem, name, mb.nrt, nblki, ni, nj, nk)
        # Add vector variables to block tree
        metaData.addVectorToBlockElem(
            blockElem, "Velocity", ["u", "v", "w"], mb.nrt, nblki, ni, nj, nk
        )

    # We add the et to the zeroth ranks mb object
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
        names = ["rho", "p", "u", "v", "w", "T"] + mb[0].speciesNames
        for nblki in range(mb.totalBlocks):
            qf.create_group(f"results_{nblki:06d}")
            ni, nj, nk = (
                int(i)
                for i in metaData.tree[0][0][nblki][0]
                .get("NumberOfElements")
                .split(" ")
            )
            extentCC = (ni - 1) * (nj - 1) * (nk - 1)
            for name in names:
                qf[f"results_{nblki:06d}"].create_dataset(
                    name, shape=(extentCC,), dtype=fdtype
                )

    # Write the data
    for blk in mb:
        # update the host views
        blk.updateHostView(["q", "Q"])

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        ng = blk.ng
        nblki = blk.nblki

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
        names = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
        for j in range(len(names)):
            dsetName = names[j]
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
