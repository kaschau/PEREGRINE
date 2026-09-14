"""The records a kernel takes for every entry of a multiBlock: each block, or
each boundary face. A block registers each array as it allocates it, into its
own slot, so the block table grows as the blocks come and is complete the
moment the last one is; the face table is filled from faces that already have
theirs."""

import ctypes

from .abi import Dims, Range, View


class Table:
    def __init__(self, ng=None):
        # the halo depth, one for the case, known once every kernel of it is
        self.ng = ng
        self.count = 0
        # each column as it is written, and as the kernels read it
        self._columns = {}
        self._arrays = {}
        self._ranges = {}

    def __len__(self):
        return self.count

    def _set(self, name, ctype, index, value):
        """One entry's value in a named column, which grows to hold it."""
        self.count = max(self.count, index + 1)
        ctype, column = self._columns.setdefault(name, (ctype, []))
        column.extend(ctype() for _ in range(index + 1 - len(column)))
        column[index] = value
        self._arrays.pop(name, None)

    def _array(self, name):
        """A column as the kernel takes it, rebuilt after any write to it."""
        if name not in self._columns:
            raise KeyError(f"no entry of this table has {name}")
        if name not in self._arrays:
            ctype, column = self._columns[name]
            array = (ctype * self.count)()
            for index, value in enumerate(column):
                array[index] = value
            self._arrays[name] = array
        return self._arrays[name]

    def register(self, index, name, array):
        """One entry's record for a named array."""
        self._set(name, View, index, View.of(array))

    def setInt(self, index, name, value):
        """One entry's value of a named per-entry integer."""
        self._set(name, ctypes.c_int, index, value)

    def setDims(self, index, blk):
        """One entry's block shape."""
        self._set("dims", Dims, index, Dims.of(blk))
        # the ranges follow the shapes
        self._ranges.clear()

    @classmethod
    def ofFaces(cls, faces):
        """A table over boundary faces: each one's block arrays, its own
        values, and which side of its block it is."""
        table = cls(faces[0].blk.ng if faces else 0)
        for index, face in enumerate(faces):
            blk = face.blk
            table.setDims(index, blk)
            table.setInt(index, "nface", face.nface)
            # what a condition's records name, whether or not this case has it
            for name in ("q", "Q", "qh", "grads"):
                table.register(index, name, getattr(blk, name, None))
            table.register(index, "S", getattr(blk, f"{face.direction}S"))
            table.register(index, "rot", face.periodicRotMatrix)
            table.register(index, "qBcVals", face.qBcVals)
            table.register(index, "QBcVals", face.QBcVals)
        return table

    def views(self, name):
        return self._array(name)

    def ints(self, name):
        return self._array(name)

    @property
    def dims(self):
        return self._array("dims")

    def ranges(self, nface=None):
        """Each entry's cell range: the given nface on every block, or each
        face's own halo."""
        if nface not in self._ranges:
            dims = self.dims
            ranges = (Range * self.count)()
            for index in range(self.count):
                side = self.ints("nface")[index] if nface is None else nface
                ranges[index] = Range.of(dims[index], self.ng, side)
            self._ranges[nface] = ranges
        return self._ranges[nface]
