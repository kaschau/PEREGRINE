"""A block's arrays, by kind. A block array is one kind of array made for
one block: shaped by the block's extents and its own components, made on
the block's backend, and referring to the range object of its kind for
every iteration space over it."""

from ..backend.array import BaseArray
from ..ranges import CellCenterRange, CellFaceRange, NodeRange


class BaseBlockArray(BaseArray):
    """One kind of array on one block: shaped by the block's extents and its
    own components, made on the block's backend, and answering the kind's
    canonical ranges through its range object."""

    # the kind of range object this kind of array refers to
    rangeKind = None

    def __init__(self, block, name, components=(), *, dtype="float64", **rangeArgs):
        self.range = self.rangeKind(block.extents, block.ng, **rangeArgs)
        self.components = tuple(components)
        shape = self.range.fullExtents + self.components
        super().__init__(shape, block.backend, dtype, name=name)


class CellCenterArray(BaseBlockArray):
    rangeKind = CellCenterRange


class NodeArray(BaseBlockArray):
    rangeKind = NodeRange


class CellFaceArray(BaseBlockArray):
    """The values on the cell faces of one direction; the axis is given at
    the declaration."""

    rangeKind = CellFaceRange
