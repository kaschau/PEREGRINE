"""The canonical ranges of every array kind obey the law: the interior and
the six halos are disjoint, together they are `all`, and no halo edge or
corner cell is in any of them."""

import numpy as np
import pytest

from peregrinepy.ranges import CellCenterRange, CellFaceRange, NodeRange

extents, ng = (8, 7, 6), 2


def kinds():
    yield "cells", CellCenterRange(extents, ng)
    yield "nodes", NodeRange(extents, ng)
    for axis in range(3):
        yield f"faces {axis}", CellFaceRange(extents, ng, axis)


def marked(rng, ranges):
    """Marks every index a list of ranges covers; a count above one is an
    overlap."""
    hit = np.zeros(rng.fullExtents, dtype=int)
    for start, extent in ranges:
        hit[tuple(slice(s, s + e) for s, e in zip(start, extent))] += 1
    return hit


@pytest.mark.parametrize(
    "name,rng", list(kinds()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_allIsInteriorAndSixDisjointHalos(name, rng):
    hit = marked(rng, rng.all())
    assert hit.max() == 1
    assert hit.sum() == np.prod(rng.interiorExtents) + sum(
        np.prod(e) for n in range(1, 7) for _, e in rng.halo(n)
    )
    # every interior index once
    assert (marked(rng, rng.interior()) == 1).sum() == np.prod(rng.interiorExtents)


@pytest.mark.parametrize(
    "name,rng", list(kinds()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_noHaloEdgeOrCornerIsEverIterated(name, rng):
    hit = marked(rng, rng.all())
    lo, hi = ng, [n - ng for n in rng.fullExtents]
    # an index is in a halo edge or corner when it is outside the interior
    # along two or more axes
    idx = np.indices(rng.fullExtents)
    outside = sum((idx[a] < lo) | (idx[a] >= hi[a]) for a in range(3))
    assert not hit[outside >= 2].any()
    assert hit[outside == 0].all() and hit[outside == 1].all()


@pytest.mark.parametrize(
    "name,rng", list(kinds()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_fullCoversEverything(name, rng):
    assert (marked(rng, rng.full()) == 1).all()
    assert rng.fullExtents == tuple(n + 2 * ng for n in rng.interiorExtents)


def test_haloDepthAndExtents():
    rng = CellCenterRange(extents, ng)
    for nface in range(1, 7):
        for depth in (1, 2):
            ((start, extent),) = rng.halo(nface, depth)
            axis, low = (nface - 1) // 2, nface % 2 == 1
            assert extent[axis] == depth
            # against the block face proper, not the interior's far side
            assert start[axis] == (
                ng - depth if low else ng + rng.interiorExtents[axis]
            )
            layer, a, b = rng.haloExtents(nface, depth)
            assert layer == depth and (a, b) == tuple(
                e for d, e in enumerate(extent) if d != axis
            )


def test_nodesExchangeStartsOnePlaneIn():
    assert NodeRange(extents, ng).exchangeStartPlane == 1
    assert CellCenterRange(extents, ng).exchangeStartPlane == 0
    for axis in range(3):
        assert CellFaceRange(extents, ng, axis).exchangeStartPlane == 0


def test_cellFaceKindsHaveOneMoreAlongTheirAxis():
    cells = CellCenterRange(extents, ng).interiorExtents
    for axis in range(3):
        faces = CellFaceRange(extents, ng, axis).interiorExtents
        assert all(faces[a] == cells[a] + (a == axis) for a in range(3))
        for nface in (2 * axis + 1, 2 * axis + 2):
            ((start, extent),) = CellFaceRange(extents, ng, axis).blockFacePlane(nface)
            assert extent[axis] == 1
            assert start[axis] == (ng if nface % 2 == 1 else ng + faces[axis] - 1)
        with pytest.raises(AssertionError):
            CellFaceRange(extents, ng, axis).blockFacePlane(2 * ((axis + 1) % 3) + 1)
