"""The records a kernel takes for every entry of a multiBlock: each block, or
each boundary face. A block registers each array as it allocates it, into its
own slot, so the block table is complete the moment the last block is; the
face table is filled from faces that already have theirs."""

import ctypes

from .abi import Dims, Range, View


class Table:
    def __init__(self, count, ng):
        self.count = count
        # the halo depth, one for the case
        self.ng = ng
        self._views = {}
        self._ints = {}
        self._dims = (Dims * count)()
        self._ranges = {}

    def __len__(self):
        return self.count

    def register(self, index, name, array):
        """One entry's record for a named array."""
        if name not in self._views:
            self._views[name] = (View * self.count)()
        self._views[name][index] = View.of(array)

    def setInt(self, index, name, value):
        """One entry's value of a named per-entry integer."""
        if name not in self._ints:
            self._ints[name] = (ctypes.c_int * self.count)()
        self._ints[name][index] = value

    def setDims(self, index, blk):
        """One entry's block shape."""
        self._dims[index] = Dims.of(blk)
        # the ranges follow the shapes
        self._ranges.clear()

    @classmethod
    def ofFaces(cls, faces, hook):
        """A table over boundary faces: each one's block arrays, its own
        values, which side of its block it is, and which condition it has at
        this hook."""
        table = cls(len(faces), faces[0].blk.ng if faces else 0)
        areaVectors = {1: "iS", 2: "iS", 3: "jS", 4: "jS", 5: "kS", 6: "kS"}
        for index, face in enumerate(faces):
            blk = face.blk
            table.setDims(index, blk)
            table.setInt(index, "nface", face.nface)
            table.setInt(index, "kind", face.kind[hook])
            for name in blk.declared:
                table.register(index, name, getattr(blk, name))
            # what every condition's records name, whether or not this case has it
            table.register(index, "grads", getattr(blk, "grads", None))
            table.register(index, "S", getattr(blk, areaVectors[face.nface]))
            table.register(index, "rot", face.periodicRotMatrix)
            table.register(index, "qBcVals", face.qBcVals)
            table.register(index, "QBcVals", face.QBcVals)
        return table

    def views(self, name):
        if name not in self._views:
            raise KeyError(f"no entry of this table has {name}")
        return self._views[name]

    def ints(self, name):
        if name not in self._ints:
            raise KeyError(f"no entry of this table has {name}")
        return self._ints[name]

    @property
    def dims(self):
        return self._dims

    def ranges(self, nface=None):
        """Each entry's cell range: the given nface on every block, or each
        face's own halo."""
        if nface not in self._ranges:
            ranges = (Range * self.count)()
            for index in range(self.count):
                side = self._ints["nface"][index] if nface is None else nface
                ranges[index] = Range.of(self._dims[index], self.ng, side)
            self._ranges[nface] = ranges
        return self._ranges[nface]
