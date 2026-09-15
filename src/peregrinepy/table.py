"""The records a kernel takes for every entry of a multiBlock: each block, or
each boundary face. A block registers each array as it allocates it, into its
own slot, so the block table grows as the blocks come and is complete the
moment the last one is; the face table is filled from faces that already have
theirs."""

import ctypes

import numpy as np

from .abi import DeviceArray, Dims, Range, View


class Tiling(ctypes.Structure):
    """One launch over every entry: which entry each tile of `tileSize` items
    belongs to, how many items each entry has, and each entry's cells."""

    _fields_ = [
        ("entry", ctypes.c_void_p),
        ("first", ctypes.c_void_p),
        ("items", ctypes.c_void_p),
        ("cells", ctypes.c_void_p),
        ("count", ctypes.c_int),
        ("tiles", ctypes.c_int),
    ]
    tileSize = 128


class Table:
    def __init__(self, ng=None, ne=None):
        # the halo depth and equation count, one each for the case
        self.ng, self.ne = ng, ne
        self.count = 0
        # each column as python writes it, and the copy the kernels read
        self._columns = {}
        self._arrays = {}
        self._devices = {}
        # index -> the block or face an entry describes
        self.entries = {}
        self._ranges = {}
        self._tilings = {}

    def __len__(self):
        return self.count

    def _set(self, name, ctype, index, value):
        """One entry's value in a named column, which grows to hold it."""
        self.count = max(self.count, index + 1)
        ctype, column = self._columns.setdefault(name, (ctype, []))
        column.extend(ctype() for _ in range(index + 1 - len(column)))
        column[index] = value
        self._arrays.pop(name, None)
        self._devices.pop(name, None)

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

    def swap(self, a, b):
        """Two named arrays trade places on every entry: the records, and the
        entries' own attributes, so nothing is copied to rotate a register."""
        for index, blk in self.entries.items():
            x, y = getattr(blk, a), getattr(blk, b)
            setattr(blk, a, y), setattr(blk, b, x)
            self.register(index, a, y)
            self.register(index, b, x)

    def register(self, index, name, array):
        """One entry's record for a named array."""
        self._set(name, View, index, View.of(array))

    def setInt(self, index, name, value):
        """One entry's value of a named per-entry integer."""
        self._set(name, ctypes.c_int, index, value)

    def setDims(self, index, blk):
        """One entry's block shape, and the block itself, whose arrays the
        entry's records describe."""
        self.entries[index] = blk
        self._set("dims", Dims, index, Dims.of(blk))
        # the ranges follow the shapes
        self._ranges.clear()
        self._tilings.clear()

    @classmethod
    def ofFaces(cls, faces):
        """A table over boundary faces: each one's block arrays, its own
        values, and which side of its block it is."""
        table = cls(faces[0].blk.ng, faces[0].blk.ne) if faces else cls(0, 0)
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

    def device(self, name):
        """A column where the kernels run, copied when it has changed."""
        if name not in self._devices:
            self._devices[name] = DeviceArray.ofBytes(self._array(name))
        return self._devices[name].ptr

    ###########################################################################
    # The ranges kernels declare, tiled for one launch over every entry
    ###########################################################################
    def _extents(self, index):
        """The cell array extents of an entry's block."""
        d, ng = self.dims[index], self.ng
        return d.ni + 2 * ng - 1, d.nj + 2 * ng - 1, d.nk + 2 * ng - 1

    def _cellsOf(self, index, kind, arg, nface):
        """One entry's range for a kind: (start[3], extent[3])."""
        ng = self.ng
        ni, nj, nk = self._extents(index)
        if kind == "cells":
            side = self.ints("nface")[index] if nface is None else nface
            r = Range.of(self.dims[index], ng, side)
            return (r.i0, r.j0, r.k0), (r.i1 - r.i0, r.j1 - r.j0, r.k1 - r.k0)
        if kind == "interior":
            return (ng, ng, ng), (ni - 2 * ng, nj - 2 * ng, nk - 2 * ng)
        if kind == "interiorPlusOne":
            return (ng - 1,) * 3, (ni - 2 * ng + 2, nj - 2 * ng + 2, nk - 2 * ng + 2)
        if kind == "whole":
            return (0, 0, 0), (ni, nj, nk)
        if kind in ("iFaces", "jFaces", "kFaces"):
            mod = ["iFaces", "jFaces", "kFaces"].index(kind)
            extent = [ni - 2 * ng, nj - 2 * ng, nk - 2 * ng]
            extent[mod] += 1
            return (ng, ng, ng), tuple(extent)
        if kind == "facePlanes":
            # (layer, i, j) of a face's planes, layers deep
            axis = (self.ints("nface")[index] - 1) // 2
            i = nj if axis == 0 else ni
            j = nj if axis == 2 else nk
            return (0, 0, 0), (arg, i, j)
        raise ValueError(f"no range kind {kind}")

    def _components(self, spec):
        return {"1": 1, "3": 3, "ne": self.ne}[spec]

    def tiling(self, kind, arg=None, components="1", nface=None):
        """The tiling of a declared range over every entry, kept once made.
        A trade's items are its buffer planes; every other kind is a cell
        range of the entry's block."""
        key = (kind, arg, components, nface)
        if key not in self._tilings:
            count = self.count
            cells = np.zeros((count, 8), dtype=np.int32)
            if kind == "trades":
                for index in range(count):
                    buffer = self.views("buffer")[index]
                    nLayer = self.ints("nLayer")[index]
                    # the plane cells, layers deep; a thread does every
                    # component of one, and takes the extents off the buffer
                    extent = [nLayer] + list(buffer.extent[1:3])
                    cells[index, 3:6] = extent
                    cells[index, 7] = int(np.prod(extent))
            else:
                nc = self._components(components)
                for index in range(count):
                    start, extent = self._cellsOf(index, kind, arg, nface)
                    cells[index, :3] = start
                    cells[index, 3:6] = extent
                    cells[index, 6] = nc
                    cells[index, 7] = int(np.prod(extent)) * nc
            items = cells[:, 7].copy()
            perEntry = -(-items // Tiling.tileSize)
            first = np.zeros(count + 1, dtype=np.int32)
            first[1:] = np.cumsum(perEntry)
            # which entry each tile belongs to, so a team finds its own in one read
            entry = np.repeat(np.arange(count, dtype=np.int32), perEntry)
            arrays = [DeviceArray.ofBytes(a) for a in (entry, first, items, cells)]
            tiling = Tiling(*(a.ptr for a in arrays), count, int(first[-1]))
            # the arrays live as long as the tiling that points into them
            tiling.arrays = arrays
            self._tilings[key] = tiling
        return self._tilings[key]

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
