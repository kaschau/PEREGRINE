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
    if rank == 0:
        totalNiList = np.zeros((mb.totalBlocks, 3), dtype=np.int32)
        for blk in mb:
            totalNiList[blockIndex, 0] = blk.ni
            totalNiList[blockIndex, 1] = blk.nj
            totalNiList[blockIndex, 2] = blk.nk
            blockIndex += 1
    else:
        totalNiList = None

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
    if rank == 0:
        totalBlockList = [nblki for b in blocksForProcs for nblki in b]
        totalBlockList, totalNiList = (
            list(t) for t in zip(*sorted(zip(totalBlockList, totalNiList)))
        )

    # Create the xml for all the blocks
    if rank == 0:
        # Start the xdmf
        xdmf = restartMetaData(
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
            blockElem = xdmf.addBlockElem(nblki, ni, nj, nk, ng=0)

            # Add scalar variables to block tree
            names = ["rho", "p", "T"] + blk.speciesNames

            for name in names:
                xdmf.addScalarToBlockElem(blockElem, name, nblki, mb.nrt, ni, nj, nk)
            # Add vector variables to block tree
            xdmf.addVectorToBlockElem(
                blockElem, "Velocity", ["u", "v", "w"], nblki, mb.nrt, ni, nj, nk
            )

        # We add the et to the zeroth ranks mb object
        return xdmf
    else:
        return None


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

    for blk in mb:
        # update the host views
        blk.updateHostView(["q", "Q"])

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        ng = blk.ng

        fileName = f"{path}{metaData.getVarFileName(blk.nblki, mb.nrt)}"
        with h5py.File(fileName, "w") as qf:

            qf.create_group("iter")
            qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
            qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")

            dset = qf["iter"]["nrt"]
            dset[0] = blk.nrt
            dset = qf["iter"]["tme"]
            dset[0] = blk.tme

            qf.create_group("results")

            if blk.blockType == "solver":
                dsetName = "rho"
                qf["results"].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
                dset = qf["results"][dsetName]
                dset[:] = blk.array["Q"][ng:-ng, ng:-ng, ng:-ng, 0].ravel(order="F")
            names = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
            for j in range(len(names)):
                dsetName = names[j]
                qf["results"].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
                dset = qf["results"][dsetName]
                dset[:] = blk.array["q"][ng:-ng, ng:-ng, ng:-ng, j].ravel(order="F")
            # Compute the nth species here
            dsetName = blk.speciesNames[-1]
            qf["results"].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf["results"][dsetName]
            if blk.ns > 1:
                dset[:] = 1.0 - np.sum(
                    blk.array["q"][ng:-ng, ng:-ng, ng:-ng, 5::], axis=-1
                ).ravel(order="F")
            elif blk.ns == 1:
                dset[:] = 1.0

    # Update and write out xdmf
    if rank == 0:

        for grid in metaData.tree.getroot()[0][0]:
            nblki = int(grid.get("Name")[1::])
            time = grid.find("Time")
            time.set("Value", str(mb.tme))
            for var in grid.findall("Attribute"):
                name = var.get("Name")
                if name != "Velocity":
                    text = metaData.getVarFileH5Location(name, nblki, mb.nrt)
                    var[0].text = text
                else:
                    for v in var.find("DataItem").findall("DataItem"):
                        name = v.get("Name")
                        text = metaData.getVarFileH5Location(name, nblki, mb.nrt)
                        v.text = text

        metaData.saveXdmf(path)
