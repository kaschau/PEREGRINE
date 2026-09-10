"""
Extrapolating a block's halo from its own interior, so a block standing alone
still has something outside it. What a neighbor owns is overwritten by the
exchange; this is only the starting point, and what a boundary face keeps.
"""

import numpy as np


class HaloMixin:
    # every low face before every high one; the blends are order dependent
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
        out = {}
        for name in ("face", "edge", "corner"):
            m = np.zeros((shape[0] + 2 * ng, shape[1] + 2 * ng))
            if name == "face":
                m[ng : shape[0] + ng, ng : shape[1] + ng] = 1.0
            elif name == "edge":
                m[0:ng, ng : shape[1] + ng] = 1.0
                m[ng : shape[0] + ng, 0:ng] = 1.0
                m[-ng::, ng : shape[1] + ng] = 1.0
                m[ng : shape[0] + ng, -ng::] = 1.0
            else:
                m[0:ng, 0:ng] = 1.0
                m[0:ng, -ng::] = 1.0
                m[-ng::, 0:ng] = 1.0
                m[-ng::, -ng::] = 1.0
            out[name] = np.ma.make_mask(m)
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
    def _replace(cur, extrapolated, hits):
        return extrapolated

    @staticmethod
    def _average(cur, extrapolated, hits):
        """a cell two faces both reach takes the mean of what each would set"""
        return np.where(cur == 0.0, extrapolated, 0.5 * cur + 0.5 * extrapolated)

    @staticmethod
    def _runningMean(cur, extrapolated, hits):
        """a corner is reached by three faces, so keep a running mean"""
        w = hits / (hits + 1.0)
        return np.where(cur == 0.0, extrapolated, w * cur + (1.0 - w) * extrapolated)

    # faces first, then the edges between them, then the corners between those.
    # Each pass reads what the one before it wrote, and a cell off the face
    # interior is reached by more than one face, so the blends above combine them.

    def generateHalo(self):
        ng = self.ng
        extents = (self.ni, self.nj, self.nk)
        planes = ((self.nj, self.nk), (self.ni, self.nk), (self.ni, self.nj))
        masks = {nf: self._masks(planes[(nf - 1) // 2], ng) for nf in self._order}

        varis = ["x", "y", "z"]

        # faces first, then the edges between them, then the corners between
        # those. Each pass reads what the one before it wrote, and a cell off the
        # face interior is reached by more than one face, so the blends combine
        passes = (
            ("face", self._replace),
            ("edge", self._average),
            ("corner", self._runningMean),
        )

        # the halo is built from nothing, so it starts as nothing
        for var in varis:
            x = self.array[var]
            for nface in (1, 3, 5):
                self._plane(x, nface, np.s_[0:ng])[:] = 0.0
                self._plane(x, nface, np.s_[-ng::])[:] = 0.0

        for name, blend in passes:
            for var in varis:
                x = self.array[var]
                hits = np.zeros(x.shape) if blend is self._runningMean else None
                for nface in self._order:
                    mask = masks[nface][name]
                    extent = extents[(nface - 1) // 2]
                    for n in range(ng):
                        s0, s1, s2 = self._layers(nface, n, ng, extent)
                        cur = self._plane(x, nface, s0)
                        extrapolated = (
                            2.0 * self._plane(x, nface, s1)[mask]
                            - self._plane(x, nface, s2)[mask]
                        )
                        counted = None if hits is None else self._plane(hits, nface, s0)
                        cur[mask] = blend(
                            cur[mask],
                            extrapolated,
                            None if counted is None else counted[mask],
                        )
                        if counted is not None:
                            counted[mask] += 1.0

        for var in varis:
            self.updateDeviceView(var)
