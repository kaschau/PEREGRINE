"""
Writing a PEREGRINE result.

A result is one pair of files, <name>.h5 and its <name>.xmf, named from the
step or the time it was written at (q.<nrt> unless the case says), in
the case's results directory. There is no distinction between a restart and
a frame of an animation -- a run writes as many results as it is asked for,
and any one of them can be restarted from or animated through.

    q.00000042.h5
      nrt, tme                                           attributes: when this is from
      species, variables, extras                         attributes: what each block group holds
      grid                                               attribute: the grid file, relative to this one
      config                                             attribute: the case, as its yaml
      peregrine, commit, host, ranks, command, written   attributes: where it came from
      results_000000/{rho,p,u,v,w,T,<species>}           one group per block
      results_000000/<array>                             one dataset per extra

Each variable is stored (nk-1, nj-1, ni-1) over the cells of a block of the
*grid*, not of the partition that wrote it, so a rank holding a piece of a
block writes its own hyperslab and any partition can read the result back.
That is the same shape and the same reason as the grid's coordinates. A
result carries no grid of its own -- the xdmf points at the g.h5 the case was
run on for that, which is why the blocks here have to be the grid's.
"""

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
        self.speciesNames = mb.speciesNames
        self.writesRho = "Q" in mb.arrays
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

    @property
    def dataNames(self):
        """Every variable this writer puts in the file, in q's order."""
        names = ["p", "u", "v", "w", "T"] + self.speciesNames
        if self.writesRho:
            names.insert(0, "rho")
        return names

    def _gatherExtraShapes(self, mb):
        """The shape past the cells of each extra array, from whichever rank
        holds a block."""
        mine = (
            {n: mb.blocks[0].shapeOf(n)[3:] for n in self.extras} if mb.blocks else {}
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
        names = self.dataNames

        qf = self._openCollective(self.h5FileName)
        self._stamp(qf)
        qf.attrs["nrt"], qf.attrs["tme"] = mb.nrt, mb.tme
        qf.attrs["species"] = np.array(self.speciesNames, dtype="S")
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

        # one snapshot of each block's state for the whole write: the
        # primitive vector, the density alone out of Q when there is one, and
        # the extras
        self._host = {
            id(blk): {"prims": blk.primitives()}
            | {n: getattr(blk, n).get() for n in self.extras}
            for blk in mb.blocks
        }
        if self.writesRho:
            for blk in mb.blocks:
                self._host[id(blk)]["rho"] = blk.Q.get(component=0)

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
        array, j = self._sourceFor(blk, name)
        whole = self.fileOrder(array)
        count = (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        # a variable is one component of q, or a field of its own
        picked = blk.interior + (j,) if j is not None else blk.interior
        if whole is None:
            # a CPU build's arrays are not in file order, so gather them first
            whole = np.ascontiguousarray(array[picked].T)
            sourceSel = ((0, 0, 0), count)
        elif j is None:
            sourceSel = ((ng, ng, ng), count)
        else:
            sourceSel = ((j, ng, ng, ng), (1,) + count)
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
        whole = self.fileOrder(array)
        count = comps + (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        if whole is None:
            whole = np.ascontiguousarray(array[blk.interior].T)
            sourceSel = ((0,) * len(count), count)
        else:
            sourceSel = ((0,) * len(comps) + (ng, ng, ng), count)
        destStart = (0,) * len(comps) + self._destStart(blk)
        self._writeSlab(dset, whole, sourceSel, (destStart, count))

    def _sourceFor(self, blk, name):
        """Where a named variable comes from: a component of the primitive
        vector, or a field of its own with no component."""
        held = self._host[id(blk)]
        q = held["prims"]
        if name == "rho":
            return held["rho"], None
        if name == blk.speciesNames[-1]:
            if blk.ns == 1:
                # a single species is all of it, and is not stored in q
                return np.ones(q.shape[:3], order="F"), None
            # the nth species is whatever the others leave
            return np.asfortranarray(1.0 - np.sum(q[..., 5:], axis=-1)), None
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
        if self.writesRho:
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
