r"""A block with coordinates: its nodes, its cell centers, and every metric a
physics may ask for, worked out from the nodes only when asked.

# The i,j,k block coordinate directions are \Xi (E), \Eta (N), and \Zeta (C)
#
#                  2  o--------------------------o  3
#                     |\                         |\
#                     | \                        | \
#                     |  \                       |  \
#                     |   \                      |   \
#                     |    \ 6                   |    \
#                     |     o--------------------|---- o 7
#                     |     |                    |     |
#                     |     |                    |     |
#                     |     |                    |     |
#                     |     |                    |     |
#   ^ j,N          1  o-----|--------------------o  4  |
#   |                  \    |                     \    |
#   |                   \   |                      \   |
#   |                    \  |                       \  |
#   o-----> i,E           \ |                        \ |
#    \                     \|                         \|
#     \                     o------------------------- o
#      v  k,C             5                              8
#
"""

import numpy as np

from .arrays import CellCenterArray, CellFaceArray
from .gridFace import gridFace
from .topologyBlock import topologyBlock


def metric(kind, components=(), **rangeArgs):
    """Marks a maker method as a metric of this array kind: what a grid
    declares for it, before there are nodes to make it from."""

    def mark(maker):
        maker.arrayKind = (kind, components, rangeArgs)
        return maker

    return mark


class gridBlock(topologyBlock):
    """A block with coordinates and the arrays that follow. Each array the
    multiBlock declared is stored as blk.<name>, made on the multiBlock's
    backend at setExtents; a block declares nothing of its own. The
    metrics are made from the nodes, each by its own method, only the
    ones declared."""

    # every low face before every high one; the halo blend is order dependent
    _faceOrder = (1, 3, 5, 2, 4, 6)

    def __init__(self, nblki, mb):
        self.ng = mb.ng
        # what the multiBlock says a block holds, and the backend its arrays
        # are made on
        self.declared = mb.arrays
        self.backend = mb.backend
        super().__init__(nblki)
        # every array is an attribute named for it, made once the extents are known
        for name in self.declared:
            setattr(self, name, None)

    def splitAlong(self, axis, cutIndex):
        """Gives what this block holds along its extents, split in two at
        the cut: the coordinates, the two halves sharing the plane they
        are cut on."""
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis] = slice(0, cutIndex + 1)
        high[axis] = slice(cutIndex, None)
        nodes = self.nodes.get()
        return {"nodes": (nodes[tuple(low)], nodes[tuple(high)])}

    def _newFace(self, nface):
        return gridFace(nface, self.backend, self.ng)

    ###########################################################################
    # The arrays a block has
    ###########################################################################
    @property
    def extents(self):
        """Gives this block's extents, its nodes per axis: (ni, nj, nk)."""
        return (self.ni, self.nj, self.nk)

    def setExtents(self, ni, nj, nk):
        """Sizes this block, which is what gives it its arrays, zeroed: each
        declared one made for this block by its kind."""
        super().setExtents(ni, nj, nk)
        for name in self.declared:
            setattr(self, name, self._make(name))

    def _make(self, name):
        kind, components, rangeArgs = self.declared[name]
        return kind(self, name, components, **rangeArgs)

    def replace(self, name, values):
        """Puts a new array holding these values in place of the old one of
        this name: a new arrayInfo for the tables that read it."""
        array = self._make(name)
        array.set(values)
        setattr(self, name, array)

    @property
    def interior(self):
        """Gives the slice of this block's arrays that is not halo."""
        ng = self.ng
        return np.s_[ng:-ng, ng:-ng, ng:-ng] if ng else np.s_[:, :, :]

    @property
    def baseNodeSlab(self):
        """Gives where this block's nodes sit in the base block it is a
        piece of, which is how the grid and result files are indexed.
        baseSlice counts nodes inclusively."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]

    @property
    def baseCellSlab(self):
        """Gives the same, for this block's cells. A block of ni nodes spans
        ni-1 cells, so nodes i0..i1 are cells i0..i1-1."""
        if self.baseSlice is None:
            return np.s_[:, :, :]
        i0, i1, j0, j1, k0, k1 = self.baseSlice
        return np.s_[k0:k1, j0:j1, i0:i1]

    ###########################################################################
    # The metrics, from the nodes: the cell centers every grid block has,
    # and the rest when a physics declared them
    ###########################################################################
    def computeMetrics(self, names):
        """Makes the cell centers and the named metrics from the nodes,
        each metric by its own method."""
        nodes = self.nodes.get()
        self.cells.set(self._cellCenters(nodes))
        for name in names:
            getattr(self, name).set(getattr(self, f"_{name}")(nodes))

    @staticmethod
    def _corner(nodes, axes, *highs):
        """Gives the (x, y, z) of one corner of every face plane. :axes: are
        the two the plane spans, :highs: which end of each to take."""
        s = [slice(None)] * 3
        for n, high in zip(axes, highs):
            s[n] = np.s_[1:] if high else np.s_[:-1]
        return nodes[tuple(s)]

    @staticmethod
    def _cellCenters(nodes):
        """Gives every cell center: the mean of the eight corners around it."""
        return 0.125 * (
            nodes[0:-1, 0:-1, 0:-1]
            + nodes[0:-1, 0:-1, 1::]
            + nodes[0:-1, 1::, 0:-1]
            + nodes[0:-1, 1::, 1::]
            + nodes[1::, 0:-1, 0:-1]
            + nodes[1::, 0:-1, 1::]
            + nodes[1::, 1::, 0:-1]
            + nodes[1::, 1::, 1::]
        )

    def _faceCenters(self, nodes, a):
        """Gives the center of every cell face of axis :a:: the mean of its
        four corners."""
        inPlane = [n for n in range(3) if n != a]
        return 0.25 * (
            self._corner(nodes, inPlane, 0, 0)
            + self._corner(nodes, inPlane, 0, 1)
            + self._corner(nodes, inPlane, 1, 0)
            + self._corner(nodes, inPlane, 1, 1)
        )

    def _areaVectors(self, nodes, a):
        """Gives the area vector of every cell face of axis :a:: half the
        cross product of the quad's diagonals, taken in the cyclic axis
        pair so the normal points out of the low face."""
        diagonal = [(a + 1) % 3, (a + 2) % 3]
        return 0.5 * np.cross(
            self._corner(nodes, diagonal, 1, 0) - self._corner(nodes, diagonal, 0, 1),
            self._corner(nodes, diagonal, 1, 1) - self._corner(nodes, diagonal, 0, 0),
        )

    @metric(CellFaceArray, 3, axis=0)
    def _iFaces(self, nodes):
        return self._faceCenters(nodes, 0)

    @metric(CellFaceArray, 3, axis=1)
    def _jFaces(self, nodes):
        return self._faceCenters(nodes, 1)

    @metric(CellFaceArray, 3, axis=2)
    def _kFaces(self, nodes):
        return self._faceCenters(nodes, 2)

    @metric(CellFaceArray, 3, axis=0)
    def _iS(self, nodes):
        return self._areaVectors(nodes, 0)

    @metric(CellFaceArray, 3, axis=1)
    def _jS(self, nodes):
        return self._areaVectors(nodes, 1)

    @metric(CellFaceArray, 3, axis=2)
    def _kS(self, nodes):
        return self._areaVectors(nodes, 2)

    def _volumes(self, nodes):
        """Gives every cell's volume: a third of the body diagonal dotted
        with the sum of the three high faces' area vectors, floored."""
        bodyDiagonal = nodes[1::, 1::, 1::] - nodes[0:-1, 0:-1, 0:-1]
        S = [self._areaVectors(nodes, a) for a in range(3)]
        J = (
            sum(
                bodyDiagonal[..., n]
                * (S[0][1::, :, :, n] + S[1][:, 1::, :, n] + S[2][:, :, 1::, n])
                for n in range(3)
            )
            / 3.0
        )
        return np.clip(J, 1e-16, None)

    @metric(CellCenterArray)
    def _Jinv(self, nodes):
        # the kernels only ever divide by the volume
        return 1.0 / self._volumes(nodes)

    @metric(CellCenterArray, 3)
    def _dIJK(self, nodes):
        """Gives every cell's lengths along each axis, opposite face center
        to opposite face center; along an axis one cell thick, which is not
        marched in, infinite, so no step limit binds on it."""
        lengths = []
        for a in range(3):
            centers = self._faceCenters(nodes, a)
            far, near = [slice(None)] * 3, [slice(None)] * 3
            far[a], near[a] = np.s_[1:], np.s_[:-1]
            span = centers[tuple(far)] - centers[tuple(near)]
            length = np.sqrt((span**2).sum(axis=-1))
            lengths.append(
                np.full_like(length, np.inf) if self.extents[a] == 2 else length
            )
        return np.stack(lengths, axis=-1)

    @metric(CellCenterArray)
    def _cellLength(self, nodes):
        """Gives every cell's length: the smallest distance from its
        center to one of its six face centers, the finest scale it
        resolves."""
        center = self._cellCenters(nodes)
        shortest = None
        for a in range(3):
            centers = self._faceCenters(nodes, a)
            for side in (np.s_[:-1], np.s_[1:]):
                index = [slice(None)] * 3
                index[a] = side
                distance = np.sqrt(((centers[tuple(index)] - center) ** 2).sum(axis=-1))
                shortest = (
                    distance if shortest is None else np.minimum(shortest, distance)
                )
        return shortest

    @metric(CellCenterArray, (3, 3))
    def _dENCdxyz(self, nodes):
        """Gives every cell's transformation metrics, d(E, N, C)/d(x, y, z),
        second order: the inverse of the jacobian of (x, y, z) w.r.t. the
        block coordinates, each derivative the mean of the four edges
        running that way."""
        # the eight cell corners, numbered as the diagram above
        c1, c2, c3, c4, c5, c6, c7, c8 = (
            self._corner(nodes, (0, 1, 2), *highs)
            for highs in (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
                (1, 0, 1),
            )
        )
        dE = 0.25 * ((c4 - c1) + (c8 - c5) + (c3 - c2) + (c7 - c6))
        dN = 0.25 * ((c2 - c1) + (c3 - c4) + (c7 - c8) + (c6 - c5))
        dC = 0.25 * ((c5 - c1) + (c8 - c4) + (c6 - c2) + (c7 - c3))
        # the inverse of that jacobian is its adjugate over its determinant,
        # and the adjugate's rows are the cross products of the other two
        return (
            np.stack([np.cross(dN, dC), np.cross(dC, dE), np.cross(dE, dN)], axis=-2)
            / self._volumes(nodes)[..., None, None]
        )

    def faceNormals(self, axis):
        """Gives the area and unit normal of every cell face of :axis:,
        worked back from the area vector, which is the only one of the
        three stored."""
        s = getattr(self, f"{axis}S").get()
        # a degenerate face is floored, we divide by this
        area = np.maximum(np.sqrt((s**2).sum(axis=-1)), 1e-16)
        return area, [s[..., n] / area for n in range(3)]

    ###########################################################################
    # The node halo, from this block alone: what a neighbor owns is
    # overwritten by an exchange; this is the starting point, and what a
    # boundary block face keeps
    ###########################################################################
    @staticmethod
    def _plane(x, nface, index):
        """Gives the index-plane of an array normal to a block face, as a
        view."""
        axis = (nface - 1) // 2
        return x[(slice(None),) * axis + (index,)]

    @staticmethod
    def _masks(shape, ng):
        """Gives a block face's plane split into its interior, the ring of
        edges around it, and its four corners."""
        inner = (np.s_[ng : shape[0] + ng], np.s_[ng : shape[1] + ng])
        ends = (np.s_[0:ng], np.s_[-ng:])
        out = {}
        for name in ("face", "edge", "corner"):
            m = np.zeros((shape[0] + 2 * ng, shape[1] + 2 * ng), dtype=bool)
            if name == "face":
                m[inner] = True
            elif name == "edge":
                # the ring around the face interior: off one end, inside the other
                for end in ends:
                    m[end, inner[1]] = True
                    m[inner[0], end] = True
            else:
                # off both ends at once
                for a in ends:
                    for b in ends:
                        m[a, b] = True
            out[name] = m
        return out

    @staticmethod
    def _layers(nface, n, ng, extent):
        """Gives the halo layer being filled and the two layers it
        extrapolates from, marching outward from the block. A block no
        thicker than its halo cannot reach past itself, so it steps one
        sided instead."""
        if nface % 2:
            s0 = ng - n - 1
            if extent <= ng:
                return s0, s0 + 1, s0 + 2
            return s0, ng, ng + n + 1
        s0 = -ng + n
        if extent <= ng:
            return s0, s0 - 1, s0 - 2
        return s0, -ng - 1, -ng - n - 2

    @staticmethod
    def _blend(cur, extrapolated, hits):
        """Gives what a block face writes into a node some other face may
        already have written: a face interior is reached once and takes the
        extrapolation outright; an edge is reached by two faces and a
        corner by three, so each folds into a running mean."""
        w = hits / (hits + 1.0)
        return np.where(hits == 0.0, extrapolated, w * cur + (1.0 - w) * extrapolated)

    def fillHaloWithNearest(self, name):
        """Fills every halo plane of an array with the nearest interior one:
        the starting point for a state read in over the interior alone."""
        a = getattr(self, name).get()
        ng = self.ng
        for axis in range(3):
            planes = np.moveaxis(a, axis, 0)
            planes[:ng] = planes[ng]
            planes[-ng:] = planes[-ng - 1]
        getattr(self, name).set(a)

    def generateHalo(self):
        """Extrapolates this block's node halo from its interior, block face
        by block face: the faces first, then the edges between them, then
        the corners, each pass reading what the one before it wrote."""
        ng = self.ng
        extents = (self.ni, self.nj, self.nk)
        planes = ((self.nj, self.nk), (self.ni, self.nk), (self.ni, self.nj))
        masks = {nf: self._masks(planes[(nf - 1) // 2], ng) for nf in self._faceOrder}
        x = self.nodes.get()
        # the halo is built from nothing, so it starts as nothing
        for nface in (1, 3, 5):
            self._plane(x, nface, np.s_[0:ng])[:] = 0.0
            self._plane(x, nface, np.s_[-ng::])[:] = 0.0
        for name in ("face", "edge", "corner"):
            # a node is reached once, not once per coordinate
            hits = np.zeros(x.shape[:3])
            for nface in self._faceOrder:
                mask = masks[nface][name]
                extent = extents[(nface - 1) // 2]
                for n in range(ng):
                    s0, s1, s2 = self._layers(nface, n, ng, extent)
                    cur = self._plane(x, nface, s0)
                    counted = self._plane(hits, nface, s0)
                    cur[mask] = self._blend(
                        cur[mask],
                        2.0 * self._plane(x, nface, s1)[mask]
                        - self._plane(x, nface, s2)[mask],
                        counted[mask][:, None],
                    )
                    counted[mask] += 1.0
        self.nodes.set(x)

    def movePeriodicHalos(self):
        """Moves the node halo behind each periodic block face to where its
        transform puts it: the halo came from the partner, turned or moved
        or both."""
        periodic = [f for f in self.faces if f.periodicRotation is not None]
        if not periodic:
            return
        nodes = self.nodes.get()
        for face in periodic:
            R, t = face.periodicRotation, face.periodicTranslation
            h = face.halo(nodes)
            h[...] = (h.reshape(-1, 3) @ R.T + t).reshape(h.shape)
        self.nodes.set(nodes)
