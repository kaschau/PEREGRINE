import h5py
import numpy as np
from copy import deepcopy
from h5py import h5fd, h5p, h5s
from lxml import etree

from ..mpiComm.mpiUtils import getCommRankSize


class BaseWriter:

    def __init__(self, mb, path="./", precision="single"):
        self.path = path
        self.precision = precision
        self.fdtype = "float64" if precision == "double" else "float32"

        self.comm, self.rank, self.size = getCommRankSize()
        self.extents = self._gatherExtents(mb)

        self._dxpl = h5p.create(h5p.DATASET_XFER)
        self._dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

        self._tree = None

    ###########################################################################
    # What the grid is
    ###########################################################################
    def _gatherExtents(self, mb):
        """Every block's ni,nj,nk indexed by block number, on every rank."""
        mine = [(blk.nblki, blk.ni, blk.nj, blk.nk) for blk in mb]
        everyones = [blk for perRank in self.comm.allgather(mine) for blk in perRank]

        extents = np.zeros((len(everyones), 3), dtype=np.int32)
        for nblki, ni, nj, nk in everyones:
            extents[nblki] = ni, nj, nk

        return extents

    @property
    def totalBlocks(self):
        return len(self.extents)

    ###########################################################################
    # The hdf5 side
    ###########################################################################
    def _openCollective(self, fileName, mode="w"):
        return h5py.File(f"{self.path}/{fileName}", mode, driver="mpio", comm=self.comm)

    def _writeSlab(self, dset, source=None, sourceSel=None, destSel=None):
        """One collective write of one rank's slab of one dataset.

        Every rank must call this for every dataset, so a rank that owns no
        piece of this one passes source None and shows up with an empty
        selection -- that is what collective transfer requires, and skipping
        the call instead hangs the ranks that do have data.

        :source: is handed to HDF5 as-is with a memory hyperslab picking the
        slab out of it, so a block's array goes to disk without being copied
        into a contiguous temporary first.
        """
        fspace = dset.id.get_space()

        if source is None:
            fspace.select_none()
            dset.id.write(
                h5s.create(h5s.NULL),
                fspace,
                np.empty(0, dtype=dset.dtype),
                dxpl=self._dxpl,
            )
            return

        mspace = h5s.create_simple(source.shape)
        mspace.select_hyperslab(*sourceSel)
        fspace.select_hyperslab(*destSel)
        dset.id.write(mspace, fspace, source, dxpl=self._dxpl)

    @staticmethod
    def fileOrder(array):
        """A block array as the file stores it, (k, j, i). On a GPU build the
        views are LayoutLeft, so this transpose is a view of the host mirror
        and nothing is copied; on a CPU build it is not contiguous and HDF5
        cannot take it, so the caller falls back to a contiguous copy."""
        transposed = array.T
        return transposed if transposed.flags["C_CONTIGUOUS"] else None

    ###########################################################################
    # The xdmf side
    ###########################################################################
    @property
    def tree(self):
        if self._tree is None:
            self._buildTree()
        return self._tree

    @property
    def gridElem(self):
        if self._tree is None:
            self._buildTree()
        return self._gridElem

    def _buildTree(self):
        """The xdmf a reader opens this writer's files through. A writer that
        only adds to the grid file never asks for it, and a grid of many
        blocks is a lot of xml to build for nothing, so it waits to be asked."""
        self._tree = etree.Element("Xdmf")
        self._tree.set("Version", "2")
        domainElem = etree.SubElement(self._tree, "Domain")
        self._gridElem = etree.SubElement(domainElem, "Grid")
        self._gridElem.set("Name", "PEREGRINE Output")
        self._gridElem.set("GridType", "Collection")
        self._gridElem.set("CollectionType", "Spatial")

        self._buildBlockTemplate()
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            self._addBlockElem(nblki, ni, nj, nk)

    def _buildBlockTemplate(self):
        """One block's worth of xdmf, deep copied per block."""
        self.blockTemplate = etree.Element("Grid")
        self.blockTemplate.set("Name", "B#Here")

        topologyElem = etree.SubElement(self.blockTemplate, "Topology")
        topologyElem.set("TopologyType", "3DSMesh")
        topologyElem.set("NumberOfElements", "Num Elem Here")
        geometryElem = etree.SubElement(self.blockTemplate, "Geometry")
        geometryElem.set("GeometryType", "X_Y_Z")

        dataElem = etree.SubElement(geometryElem, "DataItem")
        dataElem.set("NumberType", "Float")
        dataElem.set("Dimensions", "XYZ Dims Here")
        dataElem.set("Precision", "8" if self.precision == "double" else "4")
        dataElem.set("Format", "HDF")
        dataElem.text = "grid file location here"
        for _ in range(2):
            geometryElem.append(deepcopy(dataElem))

    def _addBlockElem(self, nblki, ni, nj, nk):
        blockElem = deepcopy(self.blockTemplate)
        blockElem.set("Name", f"B{nblki:06d}")
        blockElem.find("Topology").set("NumberOfElements", f"{nk} {nj} {ni}")

        for i, coord in enumerate(["x", "y", "z"]):
            coordElem = blockElem.find("Geometry")[i]
            coordElem.set("Dimensions", f"{nk} {nj} {ni}")
            coordElem.text = self.getGridFileH5Location(coord, nblki)

        self._decorateBlockElem(blockElem, nblki, ni, nj, nk)
        self._gridElem.append(blockElem)

    def _decorateBlockElem(self, blockElem, nblki, ni, nj, nk):
        """What this kind of writer hangs on a block beyond its coordinates."""

    def getGridFileH5Location(self, coord, nblki):
        return f"{self.gridPath}/g.h5:/coordinates_{nblki:06d}/{coord}"

    def saveXdmf(self):
        if self.rank != 0:
            return
        et = etree.ElementTree(self.tree)
        et.write(
            f"{self.path}/{self.xmfFileName}",
            pretty_print=True,
            encoding="UTF-8",
            xml_declaration=True,
        )
