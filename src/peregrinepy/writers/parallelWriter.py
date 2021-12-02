# -*- coding: utf-8 -*-

import h5py
import numpy as np
from lxml import etree
from copy import deepcopy
from ..mpiComm.mpiUtils import getCommRankSize


def registerParallelXdmf(mb, path="./", gridPath="./"):

    comm, rank, size = getCommRankSize()
    # the mb with Block0 must get a list of all other block's ni,nj,nj
    myBlockList = mb.blockList
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
        totalBlockList = [nblki for b in totalBlockList for nblki in b]
        totalNiList = [ni for n in totalNiList for ni in n]
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
            dataXElem.set("NumberType", "Float")
            dataXElem.set("Dimensions", f"{nk} {nj} {ni}")
            dataXElem.set("Precision", "8")
            dataXElem.set("Format", "HDF")
            dataXElem.text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/x"

            geometryElem.append(deepcopy(dataXElem))
            geometryElem[-1].text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/y"

            geometryElem.append(deepcopy(dataXElem))
            geometryElem[-1].text = f"{gridPath}/gv.{nblki:06d}.h5:/coordinates/z"

            # Only solvers will call this
            names = ["rho", "p", "T"] + mb[0].speciesNames

            name = names[0]
            # Attributes
            attributeElem = etree.SubElement(blockElem, "Attribute")
            attributeElem.set("Name", name)
            attributeElem.set("AttributeType", "Scalar")
            attributeElem.set("Center", "Cell")
            dataResElem = etree.SubElement(attributeElem, "DataItem")
            dataResElem.set("NumberType", "Float")
            dataResElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
            dataResElem.set("Precision", "8")
            dataResElem.set("Format", "HDF")

            text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
            dataResElem.text = text

            for name in names[1::]:
                blockElem.append(deepcopy(attributeElem))
                blockElem[-1].set("Name", name)
                text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                blockElem[-1][0].text = text

            # Velocity Attributes
            attributeElem = etree.SubElement(blockElem, "Attribute")
            attributeElem.set("Name", "Velocity")
            attributeElem.set("AttributeType", "Vector")
            attributeElem.set("Center", "Cell")
            function = etree.SubElement(attributeElem, "DataItem")
            function.set("ItemType", "Function")
            function.set("Function", "JOIN($0, $1, $2)")
            function.set("Dimensions", f"{nk-1} {nj-1} {ni-1} 3")

            for name in ["u", "v", "w"]:
                dataResElem = etree.SubElement(function, "DataItem")
                dataResElem.set("NumberType", "Float")
                dataResElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
                dataResElem.set("Precision", "8")
                dataResElem.set("Format", "HDF")
                dataResElem.set("Name", name)
                text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                dataResElem.text = text

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
        # update the host views
        blk.updateHostView("q")
        blk.updateHostView("Q")

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
                if name != "Velocity":
                    text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                    var[0].text = text
                else:
                    for v in var.find("DataItem").findall("DataItem"):
                        name = v.get("Name")
                        text = f"q.{mb.nrt:08d}.{nblki:06d}.h5:/results/{name}"
                        v.text = text

        saveFile = f"{path}/q.{mb.nrt:08d}.xmf"
        et.write(saveFile, pretty_print=True, encoding="UTF-8", xml_declaration=True)
