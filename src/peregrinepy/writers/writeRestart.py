# -*- coding: utf-8 -*-

import h5py
import numpy as np
from lxml import etree
from copy import deepcopy


def writeRestart(mb, path="./", gridPath="./", precision="double"):
    """This function produces an hdf5 file from a peregrinepy.multiBlock.restart for viewing in Paraview.

    Parameters
    ----------

    mb : peregrinepy.multiBlock.restart

    filePath : str
        Path to location to write output files

    precision : str
        Options - 'single' for single precision
                  'double' for double precision

    Returns
    -------
    None

    """

    if precision == "double":
        fdtype = "float64"
    else:
        fdtype = "float32"

    xdmfElem = etree.Element("Xdmf")
    xdmfElem.set("Version", "2")

    domainElem = etree.SubElement(xdmfElem, "Domain")

    gridElem = etree.SubElement(domainElem, "Grid")
    gridElem.set("Name", "PEREGRINE Output")
    gridElem.set("GridType", "Collection")
    gridElem.set("CollectionType", "Spatial")

    for blk in mb:

        extentCC = (blk.ni - 1) * (blk.nj - 1) * (blk.nk - 1)

        if blk.blockType == "solver":
            ng = blk.ng
            writeS = np.s_[ng:-ng, ng:-ng, ng:-ng]
        else:
            ng = 1
            writeS = np.s_[:, :, :]

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
                try:
                    dset[:] = blk.array["Q"][writeS + tuple([0])].ravel(order="F")
                except TypeError:
                    # Sometime we may not have density, so just make a zero array
                    dset[:] = np.zeros(blk.array["q"][:, :, :, 0][writeS].shape).ravel(order="F")
            names = ["p", "u", "v", "w", "T"] + blk.speciesNames[0:-1]
            for j in range(len(names)):
                dsetName = names[j]
                qf["results"].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
                dset = qf["results"][dsetName]
                dset[:] = blk.array["q"][writeS + tuple([j])].ravel(order="F")
            # Compute the nth species here
            dsetName = blk.speciesNames[-1]
            qf["results"].create_dataset(dsetName, shape=(extentCC,), dtype=fdtype)
            dset = qf["results"][dsetName]
            if blk.ns > 1:
                dset[:] = 1.0 - np.sum(
                    blk.array["q"][writeS + tuple([slice(5, None, None)])], axis=-1).ravel(order="F")
            elif blk.ns == 1:
                dset[:] = 1.0

        blockElem = etree.Element("Grid")
        blockElem.set("Name", f"B{blk.nblki:06d}")

        timeElem = etree.SubElement(blockElem, "Time")
        timeElem.set("Value", str(mb.tme))

        topologyElem = etree.SubElement(blockElem, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set("NumberOfElements", f"{blk.nk} {blk.nj} {blk.ni}")

        geometryElem = etree.SubElement(blockElem, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataXElem = etree.SubElement(geometryElem, "DataItem")
        dataXElem.set("NumberType", "Float")
        dataXElem.set("Dimensions", f"{blk.nk} {blk.nj} {blk.ni}")
        dataXElem.set("Precision", "8")
        dataXElem.set("Format", "HDF")
        dataXElem.text = f"{gridPath}/gv.{blk.nblki:06d}.h5:/coordinates/x"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = f"{gridPath}/gv.{blk.nblki:06d}.h5:/coordinates/y"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = f"{gridPath}/gv.{blk.nblki:06d}.h5:/coordinates/z"

        if blk.blockType == "solver":
            names = ["rho", "p", "T"] + blk.speciesNames
        else:
            names = ["p", "T"] + blk.speciesNames
        name = names[0]
        # Attributes
        attributeElem = etree.SubElement(blockElem, "Attribute")
        attributeElem.set("Name", name)
        attributeElem.set("AttributeType", "Scalar")
        attributeElem.set("Center", "Cell")
        dataResElem = etree.SubElement(attributeElem, "DataItem")
        dataResElem.set("NumberType", "Float")
        dataResElem.set("Dimensions", f"{blk.nk-1} {blk.nj-1} {blk.ni-1}")
        dataResElem.set("Precision", "8")
        dataResElem.set("Format", "HDF")

        text = f"q.{mb.nrt:08d}.{blk.nblki:06d}.h5:/results/{name}"
        dataResElem.text = text

        for name in names[1::]:
            blockElem.append(deepcopy(attributeElem))
            blockElem[-1].set("Name", name)
            text = f"q.{mb.nrt:08d}.{blk.nblki:06d}.h5:/results/{name}"
            blockElem[-1][0].text = text

        # Velocity Attributes
        attributeElem = etree.SubElement(blockElem, "Attribute")
        attributeElem.set("Name", "Velocity")
        attributeElem.set("AttributeType", "Vector")
        attributeElem.set("Center", "Cell")
        function = etree.SubElement(attributeElem, "DataItem")
        function.set("ItemType", "Function")
        function.set("Function", "JOIN($0, $1, $2)")
        function.set("Dimensions", f"{blk.nk-1} {blk.nj-1} {blk.ni-1} 3")

        for name in ["u", "v", "w"]:
            dataResElem = etree.SubElement(function, "DataItem")
            dataResElem.set("NumberType", "Float")
            dataResElem.set("Dimensions", f"{blk.nk-1} {blk.nj-1} {blk.ni-1}")
            dataResElem.set("Precision", "8")
            dataResElem.set("Format", "HDF")
            dataResElem.set("Name", name)
            text = f"q.{mb.nrt:08d}.{blk.nblki:06d}.h5:/results/{name}"
            dataResElem.text = text

        gridElem.append(deepcopy(blockElem))

    et = etree.ElementTree(xdmfElem)
    saveFile = f"{path}/q.{mb.nrt:08d}.xmf"
    et.write(saveFile, pretty_print=True, encoding="UTF-8", xml_declaration=True)
