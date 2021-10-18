# -*- coding: utf-8 -*-
import h5py
import numpy as np
from lxml import etree
from copy import deepcopy
from ..misc import progressBar


def writeGrid(mb, path="./", precision="double", withHalo=False):
    """This function produces an hdf5 file from a peregrinepy.multiBlock.grid (or a descendant) for viewing in Paraview.
    Parameters
    ----------
    mb : peregrinepy.multiBlock.grid (or a descendant)
    file_path : str
        Path to location to write output files
    precision : str
        Options - 'single' for single precision
                  'double' for double precision
    Returns
    -------
    None
    """

    if precision == "single":
        fdtype = "<f4"
    else:
        fdtype = "<f8"

    xdmfElem = etree.Element("Xdmf")
    xdmfElem.set("Version", "2")

    domainElem = etree.SubElement(xdmfElem, "Domain")

    gridElem = etree.SubElement(domainElem, "Grid")
    gridElem.set("Name", "PEREGRINE Output")
    gridElem.set("GridType", "Collection")
    gridElem.set("CollectionType", "Spatial")

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

        with h5py.File(f"{path}/gv.{blk.nblki:06d}.h5", "w") as f:
            f.create_group("coordinates")
            f.create_group("dimensions")

            f["dimensions"].create_dataset("ni", shape=(1,), dtype="int32")
            f["dimensions"].create_dataset("nj", shape=(1,), dtype="int32")
            f["dimensions"].create_dataset("nk", shape=(1,), dtype="int32")

            dset = f["dimensions"]["ni"]
            dset[0] = blk.ni + 2 * ng
            dset = f["dimensions"]["nj"]
            dset[0] = blk.nj + 2 * ng
            dset = f["dimensions"]["nk"]
            dset[0] = blk.nk + 2 * ng

            extent = (blk.ni + 2 * ng) * (blk.nj + 2 * ng) * (blk.nk + 2 * ng)
            f["coordinates"].create_dataset("x", shape=(extent,), dtype=fdtype)
            f["coordinates"].create_dataset("y", shape=(extent,), dtype=fdtype)
            f["coordinates"].create_dataset("z", shape=(extent,), dtype=fdtype)

            dset = f["coordinates"]["x"]
            dset[:] = blk.array["x"][writeS].ravel(order="F")
            dset = f["coordinates"]["y"]
            dset[:] = blk.array["y"][writeS].ravel(order="F")
            dset = f["coordinates"]["z"]
            dset[:] = blk.array["z"][writeS].ravel(order="F")

        blockElem = etree.Element("Grid")
        blockElem.set("Name", f"B{blk.nblki:06d}")

        topologyElem = etree.SubElement(blockElem, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set(
            "NumberOfElements", f"{blk.nk+2*ng} {blk.nj+2*ng} {blk.ni+2*ng}"
        )

        geometryElem = etree.SubElement(blockElem, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataXElem = etree.SubElement(geometryElem, "DataItem")
        dataXElem.set("ItemType", "Hyperslab")
        dataXElem.set("Dimensions", f"{blk.nk+2*ng} {blk.nj+2*ng} {blk.ni+2*ng}")
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
        dataX2Elem.text = f"gv.{blk.nblki:06d}.h5:/coordinates/x"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1][1].text = f"gv.{blk.nblki:06d}.h5:/coordinates/y"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1][1].text = f"gv.{blk.nblki:06d}.h5:/coordinates/z"

        gridElem.append(deepcopy(blockElem))

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out block {blk.nblki}")

    et = etree.ElementTree(xdmfElem)
    save_file = f"{path}/gv.xmf"
    et.write(save_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
