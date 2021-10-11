# -*- coding: utf-8 -*-

import h5py
import numpy as np
from lxml import etree
from copy import deepcopy
from ..mpiComm.mpiUtils import getCommRankSize


def registerParallelXdmf(mb, path="./", gridPath="./"):

    comm, rank, size = getCommRankSize()
    # the mb with BLock0 must get a list of all other block's ni,nj,nj
    myBlockList = [blk.nblki for blk in mb]
    myNiList = [[blk.ni, blk.nj, blk.nk] for blk in mb]

    if rank == 0:
        totalBlockList = [[i for i in myBlockList]]
        totalNiList = [[i for i in myNiList]]
    else:
        totalBlockList = None
        totalNiList = None

    for sendrank in range(1, size):
        # Send block list
        if rank == sendrank:
            tag = int(f"1{rank}201")
            comm.send(myBlockList, dest=0, tag=tag)
        # recv block list
        elif rank == 0:
            tag = int(f"1{sendrank}201")
            recvBlockList = comm.recv(source=sendrank, tag=tag)
            totalBlockList.append(recvBlockList)
        else:
            pass

        # send ni list
        if rank == sendrank:
            tag = int(f"1{rank}201")
            comm.send(myNiList, dest=0, tag=tag)
        # recv ni list
        elif rank == 0:
            tag = int(f"1{sendrank}201")
            recvNiList = comm.recv(source=sendrank, tag=tag)
            totalNiList.append(recvNiList)
        else:
            pass

    # Flatten the list, then sort in block order
    if rank == 0:
        totalBlockList = [nblki for l in totalBlockList for nblki in l]
        totalNiList = [ni for l in totalNiList for ni in l]
        totalBlockList, totalNiList = (
            list(t) for t in zip(*sorted(zip(totalBlockList, totalNiList)))
        )

    # Create the xml for all the blocks
    if rank == 0:
        xdmfElem = etree.Element("Xdmf")
        xdmfElem.set("Version", "2")

        domainElem = etree.SubElement(xdmfElem, "Domain")

        gridElem = etree.SubElement(domainElem, "Grid")
        gridElem.set("Name", "PEREGRINE Output")
        gridElem.set("GridType", "Collection")
        gridElem.set("CollectionType", "Spatial")

        for nblki, n in zip(totalBlockList, totalNiList):
            ni = n[0]
            nj = n[1]
            nk = n[2]

            extent = ni * nj * nk
            extentCC = (ni - 1) * (nj - 1) * (nk - 1)

            blockElem = etree.Element("Grid")
            blockElem.set("Name", f"B{nblki:06d}")

            timeElem = etree.SubElement(blockElem, "Time")
            timeElem.set("Value", str(mb.tme))

            topologyElem = etree.SubElement(blockElem, "Topology")
            topologyElem.set("TopologyType", "3DSMesh")
            topologyElem.set("NumberOfElements", f"{nk} {nj} {ni}")

            geometryElem = etree.SubElement(blockElem, "Geometry")
            geometryElem.set("GeometryType", "X_Y_Z")

            dataXElem = etree.SubElement(geometryElem, "DataItem")
            dataXElem.set("ItemType", "Hyperslab")
            dataXElem.set("Dimensions", f"{nk} {nj} {ni}")
            dataXElem.set("Type", "HyperSlab")
            dataX1Elem = etree.SubElement(dataXElem, "DataItem")
            dataX1Elem.set("DataType", "Int")
            dataX1Elem.set("Dimensions", "3")
            dataX1Elem.set("Format", "XML")
            dataX1Elem.text = f"0 1 {extent}"
            dataX2Elem = etree.SubElement(dataXElem, "DataItem")
            dataX2Elem.set("NumberType", "Float")
            dataX2Elem.set("ItemType", "Uniform")
            dataX2Elem.set("Dimensions", f"{extent}")
            dataX2Elem.set("Precision", "4")
            dataX2Elem.set("Format", "HDF")
            dataX2Elem.text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/x"

            geometryElem.append(deepcopy(dataXElem))
            geometryElem[-1][1].text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/y"

            geometryElem.append(deepcopy(dataXElem))
            geometryElem[-1][1].text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/z"

            # Only solvers will call this
            names = ["rho", "p", "u", "v", "w", "T"] + mb[0].speciesNames

            name = names[0]
            # Attributes
            attributeElem = etree.SubElement(blockElem, "Attribute")
            attributeElem.set("Name", name)
            attributeElem.set("AttributeType", "Scalar")
            attributeElem.set("Center", "Cell")
            dataResElem = etree.SubElement(attributeElem, "DataItem")
            dataResElem.set("ItemType", "Hyperslab")
            dataResElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
            dataResElem.set("Type", "HyperSlab")
            dataRes1Elem = etree.SubElement(dataResElem, "DataItem")
            dataRes1Elem.set("DataType", "Int")
            dataRes1Elem.set("Dimensions", "3")
            dataRes1Elem.set("Format", "XML")
            dataRes1Elem.text = f"0 1 {extentCC}"
            dataRes2Elem = etree.SubElement(dataResElem, "DataItem")
            dataRes2Elem.set("NumberType", "Float")
            dataRes2Elem.set("Dimensions", f"{extentCC}")
            dataRes2Elem.set("Precision", "4")
            dataRes2Elem.set("Format", "HDF")

            text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
            dataRes2Elem.text = text

            for name in names[1::]:
                blockElem.append(deepcopy(attributeElem))
                blockElem[-1].set("Name", name)
                text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                blockElem[-1][0][1].text = text

            gridElem.append(deepcopy(blockElem))

        # We add the et to the zeroth ranks mb object
        mb.parallelXmf = etree.ElementTree(xdmfElem)
    else:
        mb.parallelXmf = None


def parallelWriteRestart(mb, path="./", gridPath="./", precision="double"):

    comm, rank, size = getCommRankSize()

    if precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    for blk in mb:

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)
        ng = blk.ng

        fileName = f"{path}/q.{mb.nrt:08d}.{blk.nblki:06d}.h5"

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

    # Write out xmf
    if rank == 0:
        et = mb.parallelXmf

        for grid in et.getroot()[0][0]:
            nblki = int(grid.get("Name")[1::])
            time = grid.find("Time")
            time.set("Value", str(mb.tme))
            time = grid.find("Time")
            for var in grid.findall("Attribute"):
                name = var.get("Name")
                text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                var[0][-1].text = text

        saveFile = f"{path}/q.{mb.nrt:08d}.xmf"
        et.write(saveFile, pretty_print=True, encoding="UTF-8", xml_declaration=True)
