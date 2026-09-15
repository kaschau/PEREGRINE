"""The entries a launch runs over -- the blocks of a case, the boundary faces
with one condition, or the trades of one variable -- and the columns of
records a kernel takes from them. Nothing is pushed in: an entry answers
column(name) with an array, an integer or its shape, or None for an array it
does not have. A column is pulled the first time a kernel asks for it and
kept where the kernels run until an entry says forget; the tiling of a range
a kernel declares is made the same way, off the entries' blocks."""

import ctypes
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .abi import null, pgCells, pgDims, pgTiling


def blockOf(entry):
    """The block an entry's cells are of: a face's or a trade's, or itself."""
    return getattr(entry, "blk", entry)


###############################################################################
# The ranges a kernel declares, one class per kind, built from the
# declaration's parameters; each is its own key, so a table keeps one tiling
# per distinct range
###############################################################################
@dataclass(frozen=True)
class BaseRange:
    """What a kernel runs over, for one entry: a start and an extent per
    axis in the entry's block, and the components a thread does. `ng` and
    `ne` mean the block's."""

    kind: ClassVar[str] = None
    components: int | str = 1

    def cells(self, entry):
        """(start, extent) in the block's cell arrays."""
        raise NotImplementedError

    def at(self, nface):
        """This range as a call narrows it; only cell centers can be."""
        return self

    @staticmethod
    def extents(blk):
        ng = blk.ng
        return tuple(n + 2 * ng - 1 for n in (blk.ni, blk.nj, blk.nk))

    def nComponents(self, blk):
        return blk.ne if self.components == "ne" else int(self.components)


@dataclass(frozen=True)
class CellCenters(BaseRange):
    """The block's cells with `halo` layers of the halo around them: none
    for the interior, ng for the whole block. Over a block face table, the
    halo cells behind each face."""

    kind: ClassVar[str] = "cellCenters"
    halo: int | str = 0

    def cells(self, entry):
        side = entry.column("nface")
        if side is not None:
            return BlockFaceHalo(self.components, side).cells(entry)
        blk = blockOf(entry)
        start = blk.ng - (blk.ng if self.halo == "ng" else int(self.halo))
        return (start,) * 3, tuple(n - 2 * start for n in self.extents(blk))

    def at(self, nface):
        # a call's side: -1 as declared, 0 the interior, 1..6 one face's halo
        if nface is None or nface == -1:
            return self
        if nface == 0:
            return CellCenters(self.components, 0)
        return BlockFaceHalo(self.components, nface)


@dataclass(frozen=True)
class BlockFaceHalo(BaseRange):
    """The halo cells behind one of a block's six faces, ng deep: the side
    given, or each entry's own."""

    kind: ClassVar[str] = "blockFaceHalo"
    side: int | None = None

    def cells(self, entry):
        blk = blockOf(entry)
        ng = blk.ng
        side = entry.column("nface") if self.side is None else self.side
        axis, low = (side - 1) // 2, side % 2 == 1
        start, extent = [0, 0, 0], list(self.extents(blk))
        start[axis] = 0 if low else extent[axis] - ng
        extent[axis] = ng
        return tuple(start), tuple(extent)


@dataclass(frozen=True)
class CellFaces(BaseRange):
    """The faces of one direction between the interior cells: one more than
    the cells along it."""

    kind: ClassVar[str] = "cellFaces"
    axis: int = 0

    def cells(self, entry):
        blk = blockOf(entry)
        ng = blk.ng
        extent = [n - 2 * ng for n in self.extents(blk)]
        extent[self.axis] += 1
        return (ng,) * 3, tuple(extent)


@dataclass(frozen=True)
class BlockFacePlanes(BaseRange):
    """(layer, a, b) of a block face's planes, `layers` deep."""

    kind: ClassVar[str] = "blockFacePlanes"
    layers: int | str = "ng"

    def cells(self, entry):
        blk = blockOf(entry)
        ni, nj, nk = self.extents(blk)
        axis = (entry.column("nface") - 1) // 2
        layers = blk.ng if self.layers == "ng" else int(self.layers)
        a = nj if axis == 0 else ni
        b = nj if axis == 2 else nk
        return (0, 0, 0), (layers, a, b)


@dataclass(frozen=True)
class BufferPlanes(BaseRange):
    """The plane cells of a trade's buffer, its layers deep; a thread does
    every component of one."""

    kind: ClassVar[str] = "bufferPlanes"

    def cells(self, entry):
        buffer = entry.column("buffer")
        return (0, 0, 0), (entry.column("nLayer"), *buffer.shape[1:3])


class Table:
    """The rows a launch runs over and the columns a kernel takes from them.
    The table knows an entry only through column(name) and the block its
    cells are of; it holds nothing but the columns and tilings it uploads,
    makes no kernels, and says nothing about the order launches run in."""

    def __init__(self, entries, tileSize, backend):
        # the list itself, not a copy: a block table is over the blocks as they come
        self.entries = entries
        # items of one entry per tile, the case's knob; and where the kernels
        # run, which is where the columns and tilings are kept
        self.tileSize = tileSize
        self.backend = backend
        self._columns = {}
        self._tilings = {}

    @property
    def count(self):
        return len(self.entries)

    def forget(self, name):
        """A column is stale, and so is any tiling read off one."""
        self._columns.pop(name, None)
        self._tilings.clear()

    def forgetAll(self):
        self._columns.clear()
        self._tilings.clear()

    def _upload(self, host):
        """The bytes of a ctypes or numpy array where the kernels run."""
        raw = np.frombuffer(host, dtype=np.uint8)
        array = self.backend.allocate(raw.shape, np.uint8)
        array.set(raw)
        return array

    @staticmethod
    def _record(value):
        """One entry's answer as the record its column holds."""
        if value is None:
            return null
        if isinstance(value, (bool, int, np.integer)):
            return ctypes.c_int(int(value))
        if isinstance(value, pgDims):
            return value
        return value.record

    def column(self, name):
        """Where the kernels find a column: one record per entry, of the kind
        the entries answer -- a record of an array (null for one an entry does
        not have), an integer, or a block's shape."""
        if name not in self._columns:
            values = [entry.column(name) for entry in self.entries]
            if all(v is None for v in values):
                raise KeyError(f"no entry of this table has {name}")
            records = [self._record(v) for v in values]
            host = (type(records[0]) * len(records))(*records)
            self._columns[name] = self._upload(host)
        return self._columns[name].ptr

    ###########################################################################
    # The ranges kernels declare, tiled for one launch over every entry
    ###########################################################################
    def tiling(self, rng):
        """The tiling of a range over every entry, kept once made: a tile is
        `tileSize` items of one entry."""
        if rng not in self._tilings:
            count = self.count
            cells = np.zeros(count, dtype=np.dtype(pgCells))
            for index, entry in enumerate(self.entries):
                start, extent = rng.cells(entry)
                nc = rng.nComponents(blockOf(entry))
                cells["start"][index] = start
                cells["extent"][index] = (*extent, nc)
                cells["n"][index] = int(np.prod(extent)) * nc
            items = cells["n"].copy()
            perEntry = -(-items // self.tileSize)
            first = np.zeros(count + 1, dtype=np.int32)
            first[1:] = np.cumsum(perEntry)
            # which entry each tile belongs to, so a team finds its own in one read
            entry = np.repeat(np.arange(count, dtype=np.int32), perEntry)
            arrays = [self._upload(a) for a in (entry, first, items, cells)]
            tiling = pgTiling(*(a.ptr for a in arrays), count, int(first[-1]))
            # the arrays live as long as the tiling that points into them
            tiling.arrays = arrays
            self._tilings[rng] = tiling
        return self._tilings[rng]
