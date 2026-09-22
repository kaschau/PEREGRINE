"""Writing a PEREGRINE result, a q.<name>.h5 and its .xmf; what it holds is
docs/files.md."""

import numpy as np
import yaml
from copy import deepcopy
from lxml import etree

from ..misc import Progress
from .baseWriter import BaseWriter


class RestartWriter(BaseWriter):
    """Writes a multiBlock's flow field as a numbered result. Bound to the
    multiBlock, so a solver builds one at startup and writes it every time it
    is asked for output."""

    def __init__(
        self,
        mb,
        path="./",
        gridPath="./",
        precision="single",
        quiet=True,
        basename="q.{n:08d}",
        extras=(),
        config=None,
    ):
        self.gridPath = gridPath
        # the case that writes, as its yaml, when there is one
        self.config = yaml.safe_dump(config.toDict()) if config is not None else ""
        # block arrays written beside the state
        self.extras = tuple(extras)
        # what a case starts from, and what this multiBlock writes
        self.primVars = list(mb.primVars)
        self.exportVars = list(mb.exportVars)
        # what a result is called, from its step n and time t; set by every write
        self.basename = basename
        self.name = basename.format(n=mb.nrt, t=mb.tme)
        super().__init__(mb, path, precision, quiet)
        # the shape each extra has past its cells, which every rank needs to
        # create the datasets
        self.extraShapes = self._gatherExtraShapes(mb)

    @property
    def h5FileName(self):
        return f"{self.name}.h5"

    @property
    def xmfFileName(self):
        return f"{self.name}.xmf"

    def getVarFileH5Location(self, varName, nblki):
        return f"{self.h5FileName}:/results_{nblki:06d}/{varName}"

    def _gatherExtraShapes(self, mb):
        """The shape past the cells of each extra array, from whichever rank
        holds a block."""
        mine = (
            {n: getattr(mb.blocks[0], n).components for n in self.extras}
            if mb.blocks
            else {}
        )
        shapes = {}
        for theirs in self.comm.allgather(mine):
            shapes.update(theirs)
        return shapes

    def _gatherExtents(self, mb):
        """Every base block's ni,nj,nk indexed by its number in the grid. A
        result is written in the grid's blocks so that the partition that
        wrote it is not baked into it."""
        mine = [
            (blk.baseNblki, blk.ni, blk.nj, blk.nk, blk.baseSlice) for blk in mb.blocks
        ]
        perRankLists = self.comm.allgather(mine)
        everyones = [b for perRank in perRankLists for b in perRank]

        # a collective write visits every dataset once per round, so there must
        # be as many rounds as the most pieces of one block any rank holds
        self.rounds = max(
            (
                np.bincount([b[0] for b in perRank]).max()
                for perRank in perRankLists
                if perRank
            ),
            default=1,
        )

        extents = np.zeros((max(b[0] for b in everyones) + 1, 3), dtype=np.int32)
        for baseNblki, ni, nj, nk, baseSlice in everyones:
            if baseSlice is None:
                extents[baseNblki] = ni, nj, nk
            else:
                # the pieces tile the block, so the far corner of the last one
                # is the block, and baseSlice counts nodes inclusively
                i1, j1, k1 = baseSlice[1], baseSlice[3], baseSlice[5]
                extents[baseNblki] = np.maximum(
                    extents[baseNblki], (i1 + 1, j1 + 1, k1 + 1)
                )

        return extents

    ###########################################################################
    # Writing
    ###########################################################################
    def write(self, mb):
        self.name = self.basename.format(n=mb.nrt, t=mb.tme)
        names = self.exportVars

        qf = self._openCollective(self.h5FileName)
        self._stamp(qf)
        qf.attrs["nrt"], qf.attrs["tme"] = mb.nrt, mb.tme
        qf.attrs["primVars"] = np.array(self.primVars, dtype="S")
        qf.attrs["variables"] = np.array(names, dtype="S")
        qf.attrs["extras"] = np.array(self.extras, dtype="S")
        qf.attrs["grid"] = f"{self.gridPath}/g.h5"
        qf.attrs["config"] = self.config

        # the file is collective, so every rank creates every block's datasets;
        # an extra is stored components first, the way the device holds it
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            resS = qf.create_group(f"results_{nblki:06d}")
            cells = (nk - 1, nj - 1, ni - 1)
            for name in names:
                resS.create_dataset(name, shape=cells, dtype=self.fdtype)
            for name in self.extras:
                shape = self.extraShapes[name][::-1] + cells
                resS.create_dataset(name, shape=shape, dtype=self.fdtype)

        # one snapshot of each block for the whole write: every export
        # variable, and the extras
        self._host = {
            id(blk): mb.exportData(blk, names)
            | {n: getattr(blk, n).get() for n in self.extras}
            for blk in mb.blocks
        }

        # which of my blocks are pieces of each block of the grid
        mine = {}
        for blk in mb.blocks:
            mine.setdefault(blk.baseNblki, []).append(blk)

        # the writes are collective, so every rank walks every dataset
        with Progress(len(self.extents), self.quiet) as bar:
            for nblki in range(len(self.extents)):
                resS = qf[f"results_{nblki:06d}"]
                for myRound in range(self.rounds):
                    pieces = mine.get(nblki, ())
                    blk = pieces[myRound] if myRound < len(pieces) else None
                    for name in names:
                        self._writeVariable(resS[name], blk, name)
                    for name in self.extras:
                        self._writeArray(resS[name], blk, name)
                bar.step(f"Writing out block {nblki}")

        qf.close()
        self._host = None

        self._refreshXdmf(mb)
        self.saveXdmf()

    @staticmethod
    def _destStart(blk):
        """Where this block's cells begin in its base block, in file order."""
        if blk.baseSlice is None:
            return (0, 0, 0)
        i0, _, j0, _, k0, _ = blk.baseSlice
        return (k0, j0, i0)

    def _writeVariable(self, dset, blk, name):
        """This rank's slab of one variable of one block, or nothing at all."""
        if blk is None:
            self._writeSlab(dset)
            return

        ng = blk.ng
        array = self._host[id(blk)][name]
        whole = self.fileOrder(array, dset.dtype)
        count = (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        if whole is None:
            # not in file order or the file's type, so gather it first
            whole = np.ascontiguousarray(array[blk.interior].T, dtype=dset.dtype)
            sourceSel = ((0, 0, 0), count)
        else:
            sourceSel = ((ng, ng, ng), count)
        self._writeSlab(dset, whole, sourceSel, (self._destStart(blk), count))

    def _writeArray(self, dset, blk, name):
        """This rank's slab of one whole block array, every component at
        once, or nothing at all."""
        if blk is None:
            self._writeSlab(dset)
            return

        ng = blk.ng
        array = self._host[id(blk)][name]
        comps = array.shape[3:][::-1]
        whole = self.fileOrder(array, dset.dtype)
        count = comps + (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        if whole is None:
            whole = np.ascontiguousarray(array[blk.interior].T, dtype=dset.dtype)
            sourceSel = ((0,) * len(count), count)
        else:
            sourceSel = ((0,) * len(comps) + (ng, ng, ng), count)
        destStart = (0,) * len(comps) + self._destStart(blk)
        self._writeSlab(dset, whole, sourceSel, (destStart, count))

    def _refreshXdmf(self, mb):
        """Point the tree at this result's file, and say when it is from."""
        for blockElem in self.gridElem:
            nblki = int(blockElem.get("Name")[1::])
            blockElem.find("Time").set("Value", str(mb.tme))
            for attributeElem in blockElem.findall("Attribute"):
                for dataItemElem in attributeElem.iter("DataItem"):
                    # a Function item holds others, it has no location of its own
                    if len(dataItemElem) > 0:
                        continue
                    varName = dataItemElem.text.split("/")[-1]
                    dataItemElem.text = self.getVarFileH5Location(varName, nblki)

    ###########################################################################
    # The xdmf a result needs beyond a grid's
    ###########################################################################
    def _buildBlockTemplate(self):
        super()._buildBlockTemplate()

        timeElem = etree.SubElement(self.blockTemplate, "Time")
        timeElem.set("Value", "0.0")

        self.scalarAttributeTemplate = etree.Element("Attribute")
        self.scalarAttributeTemplate.set("Name", "var name here")
        self.scalarAttributeTemplate.set("ScalarAttributeType", "Scalar")
        self.scalarAttributeTemplate.set("Center", "Cell")

        self.vectorAttributeTemplate = etree.Element("Attribute")
        self.vectorAttributeTemplate.set("Name", "vector name here")
        self.vectorAttributeTemplate.set("AttributeType", "Vector")
        self.vectorAttributeTemplate.set("Center", "Cell")
        functionElem = etree.SubElement(self.vectorAttributeTemplate, "DataItem")
        functionElem.set("ItemType", "Function")
        functionElem.set("Function", "JOIN($0, $1, $2)")
        functionElem.set("Dimensions", "Dimension here 3")

        self.dataItemTemplate = etree.Element("DataItem")
        self.dataItemTemplate.set("NumberType", "Float")
        self.dataItemTemplate.set("Dimensions", "var dim nums here")
        self.dataItemTemplate.set(
            "Precision", "8" if self.precision == "double" else "4"
        )
        self.dataItemTemplate.set("Format", "HDF")
        self.dataItemTemplate.text = "result file location here"

    def _decorateBlockElem(self, blockElem, nblki, ni, nj, nk):
        """Adds every export variable as a scalar, the velocity components
        as one vector."""
        for varName in self.exportVars:
            if varName not in "uvw":
                self._addScalarToBlockElem(blockElem, varName, nblki, ni, nj, nk)
        self._addVectorToBlockElem(
            blockElem, "Velocity", ["u", "v", "w"], nblki, ni, nj, nk
        )

    def _addScalarToBlockElem(self, blockElem, varName, nblki, ni, nj, nk):
        attributeElem = deepcopy(self.scalarAttributeTemplate)
        attributeElem.set("Name", varName)

        dataItemElem = deepcopy(self.dataItemTemplate)
        dataItemElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
        dataItemElem.text = self.getVarFileH5Location(varName, nblki)

        attributeElem.append(dataItemElem)
        blockElem.append(attributeElem)

    def _addVectorToBlockElem(self, blockElem, vectorName, varNames, nblki, ni, nj, nk):
        attributeElem = deepcopy(self.vectorAttributeTemplate)
        attributeElem.set("Name", vectorName)
        functionElem = attributeElem.find("DataItem")
        functionElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1} 3")

        for varName in varNames:
            dataItemElem = deepcopy(self.dataItemTemplate)
            dataItemElem.set("Dimensions", f"{nk-1} {nj-1} {ni-1}")
            dataItemElem.text = self.getVarFileH5Location(varName, nblki)
            functionElem.append(dataItemElem)

        blockElem.append(attributeElem)
