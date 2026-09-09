import itertools

import numpy as np
import peregrinepy as pg
import pytest
from peregrinepy.decomposition import (
    cutPath,
    mergeAll,
    performCutOperations,
    reorientBlock,
)

##############################################
# A cut plane through one block has to continue through every block it meets,
# so a cut is a path and not a single split. What comes out has to tile the
# block it came from, stay wired to itself, and merge back into what it was.
##############################################


def cube(mbDims=(1, 1, 1), dims=(13, 11, 9), lengths=(1, 1, 1), periodic=(False,) * 3):
    mb = pg.multiBlock.grid(int(np.prod(mbDims)))
    pg.grid.create.multiBlockCube(
        mb,
        mbDims=list(mbDims),
        dimsPerBlock=list(dims),
        lengths=list(lengths),
        periodic=list(periodic),
    )
    return mb


def connectionsAreMutual(mb):
    """Every face that names a neighbor is named back by it."""
    for blk in mb:
        for face in blk.faces:
            if face.neighbor is None:
                continue
            partner = mb.getBlock(face.neighbor).getFace(face.neighborNface)
            if partner.neighbor != blk.nblki:
                return False
    return True


def properRelabelings():
    """The 24 axis relabelings a right handed block can be stored in."""
    for perm in itertools.permutations(range(3)):
        swaps = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
        for flips in itertools.product((False, True), repeat=3):
            if (swaps + sum(flips)) % 2 == 0:
                yield list(perm), list(flips)


@pytest.mark.parametrize("nCuts", (1, 2, 3))
@pytest.mark.parametrize("axis", ("i", "j", "k"))
def test_cutTilesTheBlock(axis, nCuts):
    base = cube()
    work = cube()
    performCutOperations(work, [[0, axis, nCuts]])
    assert len(work) == nCuts + 1
    assert connectionsAreMutual(work)

    myAxis = "ijk".index(axis)
    # the cube is axis aligned, so the pieces sort along the coordinate the
    # cut axis runs on
    pieces = sorted(work, key=lambda blk: blk.array["xyz"[myAxis]].min())
    dropShared = [slice(None)] * 3
    dropShared[myAxis] = slice(1, None)
    for var in ("x", "y", "z"):
        joined = np.concatenate(
            [pieces[0].array[var]]
            + [p.array[var][tuple(dropShared)] for p in pieces[1:]],
            axis=myAxis,
        )
        assert np.array_equal(joined, base[0].array[var])


@pytest.mark.parametrize("axis", ("i", "j", "k"))
def test_mergeUndoesCut(axis):
    base = cube()
    work = cube()
    performCutOperations(work, [[0, axis, 3]])

    assert mergeAll(work) == 3
    assert len(work) == 1
    for var in ("x", "y", "z"):
        assert np.array_equal(work[0].array[var], base[0].array[var])


def test_evenlySpacedCuts():
    work = cube()
    performCutOperations(work, [[0, "j", 2]])
    # nj = 11, so the cuts land at int(11*2/3) = 7 then int(11/3) = 3
    assert sorted(blk.nj for blk in work) == [4, 4, 5]


@pytest.mark.parametrize("perm,flips", list(properRelabelings()))
def test_cutPathFollowsOrientation(perm, flips):
    mb = cube(mbDims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1))
    reorientBlock(mb, mb.getBlock(1), perm, flips)
    face = mb.getBlock(0).getFace(2)

    for myAxis, axis in enumerate("ijk"):
        path = cutPath(mb, 0, axis)
        if axis == "i":
            # block 1 is across an i face, so an i cut never reaches it
            assert path == [[0, "i", False]]
            continue
        # the cut arrives on whichever of the neighbor's axes runs along ours,
        # counting from its far end when that axis runs backwards
        theirAxis, counterAligned = face.signedAxis(face.orientation[myAxis])
        assert path == [[0, axis, False], [1, "ijk"[theirAxis], counterAligned]]


def test_cutRunsThroughPeriodic():
    mb = cube(mbDims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1), periodic=(True,) * 3)
    performCutOperations(mb, [[0, "j", 1]])

    assert len(mb) == 4
    assert connectionsAreMutual(mb)
    # both halves of a split periodic need its span and axis to find a partner
    for blk in mb:
        for face in blk.faces:
            if not face.bcType.startswith("periodic"):
                continue
            assert face.neighbor is not None
            assert face.periodicSpan is not None
            assert face.periodicAxis is not None
