"""
Moving a solution from one grid onto another.

Which blocks of the grid being interpolated from a block of the grid being
interpolated onto lies inside is a geometry question answered once; filling
that block's q from them is a reconstruction of a scattered field, and that
is the only part a kind of interpolator writes.
"""

import numpy as np
from scipy import spatial

from ..misc import progressBar


class BaseInterpolator:
    """Interpolates a multiBlock's solution onto another multiBlock's grid.

    Which blocks a block lies inside, the loop over variables, and holding the
    result to the range it came from are the same however the field is
    reconstructed, so a kind writes only fit().

    :verboseSearch: keeps looking for bounding blocks after every point of a
    block has been found in one, which is slower but catches grids that
    overlap in more places than the first hit."""

    interpolatorName = None

    def __init__(self, verboseSearch=False):
        self.verboseSearch = verboseSearch

    # whether a kind can produce a value outside the range it was given, and
    # so wants holding to it
    canOvershoot = True

    def prepare(self, fromPts, toPts):
        """Something callable as f(values) that takes a field sampled at
        :fromPts: and gives it at :toPts:. Built from the points alone, since
        every variable of a block is sampled at the same ones."""
        raise NotImplementedError

    def interpolate(self, mbFrom, mbTo):
        """Fill every block of mbTo from the blocks of mbFrom it lies in."""
        bounds = self.boundingBlocks(mbTo, mbFrom)
        for n, (blkTo, blksFrom) in enumerate(zip(mbTo, bounds), start=1):
            self.blocksToBlock(blksFrom, blkTo)
            progressBar(n, len(mbTo), f"Interpolating block {blkTo.nblki}")

        mbTo.nrt, mbTo.tme = mbFrom.nrt, mbFrom.tme
        if mbTo.ns > 1:
            mbTo.checkSpeciesSum(True)

    def blocksToBlock(self, blksFrom, blkTo):
        """
        Takes a list of blocks that encompass the blkTo parameter in space and interpolate all
        PEREGRINE data from those blocks onto the single block.

        Parameters
        ----------

        blksFrom : list
           List of peregrinepy.blocks.restartBlock

        blkTo : peregrinepy.blocks.restartBlock
           Restart block with populated coordinate data (x,y,z)

        Returns
        -------
        None
            Updates attributes of parameter blkTo.
        """
        for blk in blksFrom:
            blk.computeMetrics()
        blkTo.computeMetrics()

        # the points never change from one variable to the next, only what was
        # sampled at them, so the geometry is worked out once
        fromPts = np.concatenate(
            [blk.array["cells"].reshape(-1, 3) for blk in blksFrom]
        )
        toPts = blkTo.array["cells"].reshape(-1, 3)
        onto = self.prepare(fromPts, toPts)

        shape = blkTo.array["q"].shape[:3]
        for i in range(blksFrom[0].array["q"].shape[-1]):
            qvFrom = np.concatenate(
                [blk.array["q"][:, :, :, i].ravel() for blk in blksFrom]
            )
            qvTo = onto(qvFrom)

            if self.canOvershoot:
                # do not allow new extrema to be created
                qvTo = np.clip(qvTo, np.min(qvFrom), np.max(qvFrom))

            blkTo.array["q"][:, :, :, i] = qvTo.reshape(shape)

    def boundingBlocks(self, mbTo, mbFrom):
        """
        Compares two multiBlock grids (or descendants) and
        determines which blocks from mbFrom each
        individual block from mbTo reside in, in space.

        Parameters
        ----------

        mbFrom: peregrinepy.multiBlock.grid
           peregrinepy.multiBlock.grid (or a descendant). Must have coordinate data populated

        mbTo: peregrinepy.multiBlock.grid
           peregrinepy.multiBlock.grid (or a descendant). Must have coordinate data populated

        Returns
        -------
        bounding_blocks : list
           List of length len(mbTo) where each entry is itself a list of the blocks
           from mbFrom that each block in mbTo resides in, spatially.
        """

        mbFrom.computeMetrics()
        mbTo.computeMetrics()

        # the box each block of the from grid occupies, and its hull, neither
        # of which changes as we walk the blocks being interpolated onto
        fromBounds = {
            blk.nblki: np.stack(
                [
                    blk.array["nodes"].min(axis=(0, 1, 2)),
                    blk.array["nodes"].max(axis=(0, 1, 2)),
                ],
                axis=-1,
            )
            for blk in mbFrom
        }
        fromHulls = {blk.nblki: self.surfaceHull(blk) for blk in mbFrom}

        boundingBlocks = []
        for blkTo in mbTo:
            centers = blkTo.array["cells"].reshape(-1, 3)
            toLo, toHi = centers.min(axis=0), centers.max(axis=0)

            found = np.zeros(len(centers), dtype=bool)
            inside = []
            for blkFrom in mbFrom:
                lo, hi = fromBounds[blkFrom.nblki].T
                # a block whose box does not reach this one cannot hold any of
                # its cells, whatever its shape
                if np.any(lo > toHi) or np.any(hi < toLo):
                    continue

                hits = self.ptsInHull(fromHulls[blkFrom.nblki], centers)
                if hits.any():
                    inside.append(blkFrom)
                    found |= hits
                # every cell is accounted for, and only a verbose search cares
                # whether some other block holds them too
                if not self.verboseSearch and found.all():
                    break

            if not inside:
                raise ValueError(
                    f"block {blkTo.nblki} of the grid being interpolated onto lies"
                    " entirely outside the grid being interpolated from"
                )
            boundingBlocks.append(inside)
            progressBar(
                blkTo.nblki + 1, mbTo.nblks, f"Finding block {blkTo.nblki} bounds"
            )

        return boundingBlocks

    @staticmethod
    def surfaceHull(blk):
        """The convex hull of every node on a block's boundary. Sampling a
        handful of them instead is much cheaper to hull but wrong as soon as
        the block curves: a 170 degree annulus block's hull, taken off its
        corners and edge midpoints, excludes half of its own cells."""
        nodes = blk.array["nodes"]
        onSurface = np.zeros(nodes.shape[:3], dtype=bool)
        onSurface[[0, -1]] = True
        onSurface[:, [0, -1]] = True
        onSurface[:, :, [0, -1]] = True
        return spatial.ConvexHull(nodes[onSurface])

    @staticmethod
    def ptsInHull(hull, testPts):
        """Which of testPts lie inside a hull. A point is inside when it is on
        the near side of every face, and a face's plane is one of the hull's
        equations."""
        # a point on a face should count as in, so allow for the rounding that
        # puts it a hair outside
        tol = 1e-12 * np.linalg.norm(hull.max_bound - hull.min_bound)
        return np.all(
            testPts @ hull.equations[:, :3].T + hull.equations[:, 3] <= tol, axis=1
        )

    @classmethod
    def ptsInBlkBounds(cls, blk, testPts):
        """Which of testPts lie inside :blk:.

        Parameters
        ----------

        blk: peregrinepy.multiBlock.gridBlock (or a descendant).
             Must have coordinate data populated.

        testPts : np.array
           Numpy array of shape (numPts,3) defining (x,y,z) of each point to test

        Returns
        -------
        np.array
           Boolean array of shape (numPts) True if the point is inside the block
        """
        return cls.ptsInHull(cls.surfaceHull(blk), testPts)
