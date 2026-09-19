"""The ranges a launch may walk of a block's arrays.

A block's extents -- ni, nj, nk nodes and ng halo layers -- fix every
iteration space over it. A range object holds that arithmetic for one kind
of array: the cell centers, the nodes, or the cell faces of one direction.
It answers a named canonical range as ranges, each a (start, extent) in the
array's own indices: `interior` is the block proper; `halo(nface)` the ng
layers behind one block face, over the block face proper -- the active
halo, never its edges or corners; `all` is all anyone needs, the interior
and the six halos, so a halo edge or corner cell is never generated -- by
law nothing reads or writes one; `full` is the whole allocation, edges
and corners included, for a copy and nothing else. A range on a rank also
knows which block faces' halos a message brings: `allLocal` is `all` less
those halos -- what is settled on this rank -- what a launch does while
the message flies, the rest done once it lands. A cell-face range also has `blockFacePlane(nface)`,
the plane of cell faces lying on a block face of its axis, and
`interiorLocal`, the interior less those planes on the faces a message
brings."""


class BaseRange:
    """The geometry of one kind of array over a block of extents (ni, nj,
    nk) nodes with ng halo layers: this kind's own interior extents, and the
    canonical ranges as ranges in its indices."""

    # which interior plane a halo exchange sends first, counted in from the
    # block face: the first, for a kind whose plane on the face is not
    # shared across it
    exchangeStartPlane = 0

    def __init__(self, extents, ng, connOffRank=()):
        self.ni, self.nj, self.nk = extents
        self.ng = ng
        # the block faces connected off the rank, whose halos a message
        # brings, by face number
        self.connOffRank = frozenset(connOffRank)

    @property
    def interiorExtents(self):
        """Gives the extent per axis of the block proper in this kind's
        units: cells, nodes or cell faces."""
        raise NotImplementedError

    @property
    def fullExtents(self):
        """Gives the extent per axis of the whole allocation, the halos
        on."""
        return tuple(n + 2 * self.ng for n in self.interiorExtents)

    def full(self):
        return [((0, 0, 0), self.fullExtents)]

    def interior(self):
        return [((self.ng,) * 3, self.interiorExtents)]

    def halo(self, nface, depth=None):
        """Gives the halo slab behind one block face: :depth: layers (ng
        unless said), over the block face proper."""
        ng, depth = self.ng, self.ng if depth is None else depth
        axis, low = (nface - 1) // 2, nface % 2 == 1
        start, extent = [ng] * 3, list(self.interiorExtents)
        start[axis] = ng - depth if low else ng + self.interiorExtents[axis]
        extent[axis] = depth
        return [(tuple(start), tuple(extent))]

    def all(self):
        return self.interior() + [range for n in range(1, 7) for range in self.halo(n)]

    def allLocal(self):
        """Gives all but the halos a message brings."""
        return self.interior() + [
            range
            for n in range(1, 7)
            if n not in self.connOffRank
            for range in self.halo(n)
        ]

    def haloExtents(self, nface, depth=None):
        """Gives the halo slab's extents as its planes are laid out, the
        layer first: (depth, a, b), with a and b the block face proper's
        two extents in axis order."""
        ((start, extent),) = self.halo(nface, depth)
        axis = (nface - 1) // 2
        return (extent[axis],) + tuple(n for d, n in enumerate(extent) if d != axis)


class CellCenterRange(BaseRange):
    """The cell centers: one fewer than the nodes along each axis."""

    @property
    def interiorExtents(self):
        return (self.ni - 1, self.nj - 1, self.nk - 1)


class NodeRange(BaseRange):
    """The grid nodes. The plane on a block face is held by both blocks
    across it, so an exchange starts one plane in."""

    exchangeStartPlane = 1

    @property
    def interiorExtents(self):
        return (self.ni, self.nj, self.nk)


class CellFaceRange(BaseRange):
    """The cell faces of one direction: as many as the nodes along the axis,
    the cells across it."""

    def __init__(self, extents, ng, axis, connOffRank=()):
        super().__init__(extents, ng, connOffRank)
        self.axis = axis

    @property
    def interiorExtents(self):
        extents = [self.ni - 1, self.nj - 1, self.nk - 1]
        extents[self.axis] += 1
        return tuple(extents)

    def interiorLocal(self):
        """Gives the interior faces less the planes on the block faces of
        this axis whose halos a message brings."""
        start, extent = [self.ng] * 3, list(self.interiorExtents)
        for nface in self.connOffRank:
            axis, low = (nface - 1) // 2, nface % 2 == 1
            if axis == self.axis:
                extent[axis] -= 1
                start[axis] += low
        return [(tuple(start), tuple(extent))]

    def blockFacePlane(self, nface):
        """Gives the plane of cell faces lying on a block face of this
        axis, over the block face proper."""
        axis, low = (nface - 1) // 2, nface % 2 == 1
        assert axis == self.axis, (nface, self.axis)
        start, extent = [self.ng] * 3, list(self.interiorExtents)
        start[axis] = self.ng if low else self.ng + self.interiorExtents[axis] - 1
        extent[axis] = 1
        return [(tuple(start), tuple(extent))]
