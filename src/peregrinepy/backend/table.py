"""A table: the entries a launch runs over -- the blocks of a rank, or the
block faces of its blocks -- with their arrays' arrayInfos uploaded once per
name,
and the tilings of the ranges launched over them.

An array name is a column: one arrayInfo per entry, stacked in entry order
where the kernels run; an entry answers tableValue(name) with the array, an
integer, a block's dims, or None for an array it does not have. A tiling
is a named list of ranges, each some entry's (start, extent) of cells, cut
into tiles of the backend's size for that kind of item; a launch hands a
kernel one tiling and the columns it names. The table makes no kernels
and says nothing about the order launches run in."""

import ctypes

import numpy as np

from .abi import noArray, pgRange, pgDims, pgTiling


class ArrayTable:
    """The entries, their arrays' arrayInfos by name, and the tilings over
    them."""

    def __init__(self, entries, backend):
        # the list itself, not a copy: a block table is over the blocks as they come
        self.entries = entries
        # where the kernels run, which is where the columns and tilings are
        # kept, and whose tiles say how many items of one range a team does
        self.backend = backend
        self._columns = {}
        self.tilings = {}

    @property
    def count(self):
        return len(self.entries)

    def forget(self, name):
        """Drops a column that went stale: an entry's array of that name is
        another array now."""
        self._columns.pop(name, None)

    def _upload(self, host):
        """Puts the bytes of a ctypes or numpy array where the kernels
        run."""
        raw = np.frombuffer(host, dtype=np.uint8)
        array = self.backend.allocate(raw.shape, np.uint8)
        array.set(raw)
        return array

    @staticmethod
    def _infoOf(value):
        """Turns one entry's answer into the arrayInfo the kernels read."""
        if value is None:
            return noArray
        if isinstance(value, (bool, int, np.integer)):
            return ctypes.c_int(int(value))
        if isinstance(value, pgDims):
            return value
        return value.info

    def arrayInfos(self, name):
        """Gives where the kernels find the array of this name: one
        arrayInfo per entry, stacked in entry order, of the kind the
        entries answer, uploaded the first time it is asked for."""
        if name not in self._columns:
            values = [entry.tableValue(name) for entry in self.entries]
            if all(v is None for v in values):
                raise KeyError(f"no entry of this table has {name}")
            arrayInfos = [self._infoOf(v) for v in values]
            host = (type(arrayInfos[0]) * len(arrayInfos))(*arrayInfos)
            self._columns[name] = self._upload(host)
        return self._columns[name].ptr

    def tiling(self, kernel, rangeName):
        """Gives the tiling of a kernel's items over the named range of
        every entry -- full, all, interior -- made once and kept."""
        key = (
            kernel.items,
            getattr(kernel, "direction", None),
            kernel.components,
            rangeName,
        )
        if key not in self.tilings:
            ranges = [
                (n, start, extent, kernel.components)
                for n, entry in enumerate(self.entries)
                for start, extent in getattr(kernel.rangeOf(entry), rangeName)()
            ]
            self._tile(key, ranges, kernel.tileKind)
        return self.tilings[key]

    def tilingOver(self, key, entries, rangesOf, items="cells"):
        """Gives the tiling under :key: over a subset of the entries, each
        contributing the ranges :rangesOf:(entry) gives, made once and
        kept."""
        if key not in self.tilings:
            index = {id(e): n for n, e in enumerate(self.entries)}
            ranges = [
                (index[id(e)], start, extent, 1)
                for e in entries
                for start, extent in rangesOf(e)
            ]
            self._tile(key, ranges, items)
        return self.tilings[key]

    def _tile(self, key, ranges, items):
        """Makes and keeps the tiling under :key: of :ranges:, each (entry
        index, start, extent, components): a tile is as many of one range's
        items as the backend's knob for :items: (cells or elements) says."""
        tile = self.backend.tiles[items]
        count = len(ranges)
        table = np.zeros(count, dtype=np.dtype(pgRange))
        for index, (entry, start, extent, components) in enumerate(ranges):
            table["start"][index] = start
            table["extent"][index] = (*extent, components)
            table["n"][index] = int(np.prod(extent)) * components
            table["entry"][index] = entry
        perRange = -(-table["n"] // tile)
        first = np.zeros(count + 1, dtype=np.int32)
        first[1:] = np.cumsum(perRange)
        # which range each tile belongs to, so a team finds its own in one read
        ofTile = np.repeat(np.arange(count, dtype=np.int32), perRange)
        arrays = [self._upload(a) for a in (ofTile, first, table["n"].copy(), table)]
        tiling = pgTiling(*(a.ptr for a in arrays), count, int(first[-1]), tile)
        # the arrays live as long as the tiling that points into them
        tiling.arrays = arrays
        tiling.tileKind = items
        self.tilings[key] = tiling
        return tiling
