"""
Writing a PEREGRINE result.

A result is one pair of files, <name>.h5 and its <name>.xmf, named from the
step or the time it was written at (q.<nrt> unless the case says), in
the case's results directory. There is no distinction between a restart and
a frame of an animation -- a run writes as many results as it is asked for,
and any one of them can be restarted from or animated through.

    q.00000042.h5
      iter/{nrt,tme}                                     when this is from
      results_000000/{rho,p,u,v,w,T,<species>}           one group per block

Each variable is stored (nk-1, nj-1, ni-1) over the cells of a block of the
*grid*, not of the partition that wrote it, so a rank holding a piece of a
block writes its own hyperslab and any partition can read the result back.
That is the same shape and the same reason as the grid's coordinates. A
result carries no grid of its own -- the xdmf points at the g.h5 the case was
run on for that, which is why the blocks here have to be the grid's.
"""

import numpy as np
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
    ):
        self.gridPath = gridPath
        self.speciesNames = mb.speciesNames
        self.hasConservatives = mb.hasConservatives
        # what a result is called, from its step n and time t; set by every write
        self.basename = basename
        self.name = basename.format(n=mb.nrt, t=mb.tme)
        super().__init__(mb, path, precision, quiet)

    @property
    def h5FileName(self):
        return f"{self.name}.h5"

    @property
    def xmfFileName(self):
        return f"{self.name}.xmf"

    def getVarFileH5Location(self, varName, nblki):
        return f"{self.h5FileName}:/results_{nblki:06d}/{varName}"

    @property
    def dataNames(self):
        """Every variable this writer puts in the file, in q's order."""
        names = ["p", "u", "v", "w", "T"] + self.speciesNames
        if self.hasConservatives:
            names.insert(0, "rho")
        return names

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
        names = self.dataNames

        qf = self._openCollective(self.h5FileName)
        qf.create_group("iter")
        qf["iter"].create_dataset("nrt", shape=(1,), dtype="int32")
        qf["iter"].create_dataset("tme", shape=(1,), dtype="float64")
        if self.rank == 0:
            qf["iter"]["nrt"][0] = mb.nrt
            qf["iter"]["tme"][0] = mb.tme

        # the file is collective, so every rank creates every block's datasets
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            resS = qf.create_group(f"results_{nblki:06d}")
            for name in names:
                resS.create_dataset(
                    name, shape=(nk - 1, nj - 1, ni - 1), dtype=self.fdtype
                )

        # one snapshot of each block's state for the whole write
        self._host = {
            id(blk): (blk.hostCopy("q"), blk.hostCopy("Q")) for blk in mb.blocks
        }

        # which of my blocks are pieces of each block of the grid
        mine = {}
        for blk in mb.blocks:
            mine.setdefault(blk.baseNblki, []).append(blk)

        # the writes are collective, so every rank walks every dataset
        with Progress(len(self.extents), self.quiet) as bar:
            for nblki in range(len(self.extents)):
                resS = qf[f"results_{nblki:06d}"]
                for name in names:
                    for myRound in range(self.rounds):
                        pieces = mine.get(nblki, ())
                        blk = pieces[myRound] if myRound < len(pieces) else None
                        self._writeVariable(resS[name], blk, name)
                bar.step(f"Writing out block {nblki}")

        qf.close()
        self._host = None

        self._refreshXdmf(mb)
        self.saveXdmf()

    def _writeVariable(self, dset, blk, name):
        """This rank's slab of one variable of one block, or nothing at all."""
        if blk is None:
            self._writeSlab(dset)
            return

        ng = blk.ng
        array, j = self._sourceFor(blk, name)
        whole = self.fileOrder(array)
        count = (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        if whole is None:
            # a CPU build's arrays are not in file order, so gather them first
            whole = np.ascontiguousarray(array[blk.interior + (j,)].T)
            sourceSel = ((0, 0, 0), count)
        else:
            sourceSel = ((j, ng, ng, ng), (1,) + count)

        if blk.baseSlice is None:
            destStart = (0, 0, 0)
        else:
            i0, _, j0, _, k0, _ = blk.baseSlice
            destStart = (k0, j0, i0)

        self._writeSlab(dset, whole, sourceSel, (destStart, count))

    def _sourceFor(self, blk, name):
        """Which array and component of it a named variable comes from."""
        q, Q = self._host[id(blk)]
        if name == "rho":
            return Q, 0

        if name == blk.speciesNames[-1]:
            if blk.ns == 1:
                # a single species is all of it, and is not stored in q
                return np.ones(q.shape[:3] + (1,)), 0
            # the nth species is whatever the others leave
            left = 1.0 - np.sum(q[..., 5:], axis=-1)
            return left[..., np.newaxis], 0

        return q, (["p", "u", "v", "w", "T"] + blk.speciesNames).index(name)

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
        scalars = ["p", "T"] + self.speciesNames
        if self.hasConservatives:
            scalars.insert(0, "rho")

        for varName in scalars:
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
