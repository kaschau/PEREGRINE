# -*- coding: utf-8 -*-
import h5py
import numpy as np
from lxml import etree
from copy import deepcopy
from ..misc import progressBar


class gridXdmf:
    def __init__(self, path, precision, lump):
        self.lump = lump
        self.path = path

        # This is the main xdmf object
        self.tree = etree.Element("Xdmf")
        self.tree.set("Version", "2")

        self.domainElem = etree.SubElement(self.tree, "Domain")
        self.gridElem = etree.SubElement(self.domainElem, "Grid")
        self.gridElem.set("Name", "PEREGRINE Output")
        self.gridElem.set("GridType", "Collection")
        self.gridElem.set("CollectionType", "Spatial")

        # This is a template of an individual block
        self.blockTemplate = etree.Element("Grid")
        self.blockTemplate.set("Name", "B#Here")

        topologyElem = etree.SubElement(self.blockTemplate, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set("NumberOfElements", "Num Elem Here")
        geometryElem = etree.SubElement(self.blockTemplate, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataXElem = etree.SubElement(geometryElem, "DataItem")
        dataXElem.set("NumberType", "Float")
        dataXElem.set("Dimensions", "XYZ Dims Here")
        dataXElem.set("Precision", "8" if precision == "double" else "4")
        dataXElem.set("Format", "HDF")

        dataXElem.text = "gridFile location:/coordinate/x"
        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/y"

        geometryElem.append(deepcopy(dataXElem))
        geometryElem[-1].text = "gridFile location:/coordinates/z"

    def addBlock(self, nblki, ni, nj, nk, ng):

        block = deepcopy(self.blockTemplate)
        block.set("Name", f"B{nblki:06d}")
        topo = block.find("Topology")
        topo.set("NumberOfElements", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")

        for coord, i in zip(["x", "y", "z"], [0, 1, 2]):
            X = block.find("Geometry")[i]
            X.set("Dimensions", f"{nk+2*ng} {nj+2*ng} {ni+2*ng}")
            X.text = self.getGridFileLocation(nblki) + coord

        self.gridElem.append(deepcopy(block))

    def getGridFileLocation(self, nblki):
        if self.lump:
            return f"./grid.h5:/coordinates_{nblki:06d}/"
        else:
            return f"./g.{nblki:06d}.h5:/coordinates_{nblki:06d}/"


def writeGrid(mb, path="./", precision="double", withHalo=False, lump=False):
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

    # Start the xdmf tree
    xdmfTree = gridXdmf(path, precision, lump)

    # If we are lumping the files, open it here
    if lump:
        f = h5py.File(f"{path}/grid.h5", "w")

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

        nblkiS = f"{blk.nblki:06d}"
        coordS = "coordinates_" + nblkiS
        dimS = "dimensions_" + nblkiS

        # If we are doing serial output, open the file here
        if not lump:
            f = h5py.File(f"{path}/g.{nblkiS}.h5", "w")

        f.create_group(coordS)
        f.create_group(dimS)

        f[dimS].create_dataset("ni", shape=(1,), dtype="int32")
        f[dimS].create_dataset("nj", shape=(1,), dtype="int32")
        f[dimS].create_dataset("nk", shape=(1,), dtype="int32")

        dset = f[dimS]["ni"]
        dset[0] = blk.ni + 2 * ng
        dset = f[dimS]["nj"]
        dset[0] = blk.nj + 2 * ng
        dset = f[dimS]["nk"]
        dset[0] = blk.nk + 2 * ng

        extent = (blk.ni + 2 * ng) * (blk.nj + 2 * ng) * (blk.nk + 2 * ng)
        f[coordS].create_dataset("x", shape=(extent,), dtype=fdtype)
        f[coordS].create_dataset("y", shape=(extent,), dtype=fdtype)
        f[coordS].create_dataset("z", shape=(extent,), dtype=fdtype)

        dset = f[coordS]["x"]
        dset[:] = blk.array["x"][writeS].ravel(order="F")
        dset = f[coordS]["y"]
        dset[:] = blk.array["y"][writeS].ravel(order="F")
        dset = f[coordS]["z"]
        dset[:] = blk.array["z"][writeS].ravel(order="F")

        # Add block to xdmf tree
        xdmfTree.addBlock(blk.nblki, blk.ni, blk.nj, blk.nk, ng)

        if mb.mbType in ["grid", "restart"]:
            progressBar(blk.nblki + 1, len(mb), f"Writing out gridBlock {blk.nblki}")

    et = etree.ElementTree(xdmfTree.tree)
    save_file = f"{path}/g.xmf"
    et.write(save_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
