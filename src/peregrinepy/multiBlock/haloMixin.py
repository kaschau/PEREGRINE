"""
Extrapolating a block's halo from its own interior, so a block standing alone
still has something outside it. What a neighbor owns is overwritten by the
exchange; this is only the starting point, and what a boundary face keeps.
"""

import numpy as np


class HaloMixin:
    # every low face before every high one; the blend is order dependent
    _order = (1, 3, 5, 2, 4, 6)

    @staticmethod
    def _plane(x, nface, index):
        """The index-plane of an array normal to a face, as a view."""
        axis = (nface - 1) // 2
        return x[(slice(None),) * axis + (index,)]

    @staticmethod
    def _masks(shape, ng):
        """A face plane split into its interior, the ring of edges around it, and
        its four corners."""
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
        """The halo layer being filled and the two layers it extrapolates from,
        marching outward from the block. A block no thicker than its halo cannot
        reach past itself, so it steps one sided instead."""
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
        """What a face writes into a cell some other face may already have
        written. A face interior is reached once and takes the extrapolation
        outright; an edge is reached by two faces and a corner by three, so
        each one folds into a running mean of what came before it."""
        w = hits / (hits + 1.0)
        return np.where(hits == 0.0, extrapolated, w * cur + (1.0 - w) * extrapolated)

    def generateHalo(self):
        ng = self.ng
        extents = (self.ni, self.nj, self.nk)
        planes = ((self.nj, self.nk), (self.ni, self.nk), (self.ni, self.nj))
        masks = {nf: self._masks(planes[(nf - 1) // 2], ng) for nf in self._order}

        x = self.nodes.get()

        # the halo is built from nothing, so it starts as nothing
        for nface in (1, 3, 5):
            self._plane(x, nface, np.s_[0:ng])[:] = 0.0
            self._plane(x, nface, np.s_[-ng::])[:] = 0.0

        # faces first, then the edges between them, then the corners between
        # those, since each pass reads what the one before it wrote
        for name in ("face", "edge", "corner"):
            # a node is reached once, not once per coordinate
            hits = np.zeros(x.shape[:3])
            for nface in self._order:
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
